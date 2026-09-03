package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/pion/ice/v4"
	"github.com/pion/interceptor"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

const (
	msgMicOpus = byte(1)
	msgTTSOpus = byte(2)
	msgClear   = byte(3)
)

type session struct {
	id       string
	mu       sync.RWMutex
	writeMu  sync.Mutex
	pc       *webrtc.PeerConnection
	mediaWS  *websocket.Conn
	outTrack *webrtc.TrackLocalStaticSample
	out      chan []byte
}

func (s *session) sendMic(payload []byte) {
	s.mu.RLock()
	ws := s.mediaWS
	s.mu.RUnlock()
	if ws == nil {
		return
	}
	msg := append([]byte{msgMicOpus}, payload...)
	s.writeMu.Lock()
	err := ws.WriteMessage(websocket.BinaryMessage, msg)
	s.writeMu.Unlock()
	if err != nil {
		log.Printf("[%s] media send: %v", s.id, err)
	}
}

func (s *session) clearOutput() {
	for {
		select {
		case <-s.out:
		default:
			return
		}
	}
}

var (
	sessions   = map[string]*session{}
	sessionsMu sync.Mutex
	upgrader   = websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	rtcAPI     *webrtc.API
)

func getSession(id string) *session {
	sessionsMu.Lock()
	defer sessionsMu.Unlock()
	if s := sessions[id]; s != nil {
		return s
	}
	s := &session{id: id, out: make(chan []byte, 12)}
	sessions[id] = s
	return s
}

func newPeer(s *session) (*webrtc.PeerConnection, error) {
	pc, err := rtcAPI.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		return nil, err
	}
	outTrack, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2},
		"voicemem-tts", "voicemem")
	if err != nil {
		pc.Close()
		return nil, err
	}
	sender, err := pc.AddTrack(outTrack)
	if err != nil {
		pc.Close()
		return nil, err
	}
	s.mu.Lock()
	s.pc, s.outTrack = pc, outTrack
	s.mu.Unlock()
	go func() {
		b := make([]byte, 1500)
		for {
			if _, _, e := sender.Read(b); e != nil {
				return
			}
		}
	}()
	go func() {
		for payload := range s.out {
			if err := outTrack.WriteSample(media.Sample{Data: payload, Duration: 20 * time.Millisecond}); err != nil {
				log.Printf("[%s] output: %v", s.id, err)
			}
		}
	}()
	pc.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		if track.Kind() != webrtc.RTPCodecTypeAudio {
			return
		}
		log.Printf("[%s] microphone track", s.id)
		for {
			pkt, _, e := track.ReadRTP()
			if e != nil {
				return
			}
			s.sendMic(pkt.Payload)
		}
	})
	pc.OnConnectionStateChange(func(st webrtc.PeerConnectionState) {
		log.Printf("[%s] peer=%s", s.id, st.String())
	})
	return pc, nil
}

type offerBody struct {
	SDP  string `json:"sdp"`
	Type string `json:"type"`
}

func offerHandler(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/offer/")
	var body offerBody
	if id == "" || json.NewDecoder(r.Body).Decode(&body) != nil {
		http.Error(w, "bad offer", 400)
		return
	}
	s := getSession(id)
	pc, err := newPeer(s)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	if err = pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeOffer, SDP: body.SDP}); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	gather := webrtc.GatheringCompletePromise(pc)
	if err = pc.SetLocalDescription(answer); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	<-gather
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(offerBody{SDP: pc.LocalDescription().SDP, Type: "answer"})
}

func mediaHandler(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/media/")
	ws, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	s := getSession(id)
	s.mu.Lock()
	if s.mediaWS != nil {
		s.mediaWS.Close()
	}
	s.mediaWS = ws
	s.mu.Unlock()
	log.Printf("[%s] python media connected", id)
	defer func() {
		s.mu.Lock()
		if s.mediaWS == ws {
			s.mediaWS = nil
		}
		s.mu.Unlock()
		ws.Close()
	}()
	for {
		mt, data, e := ws.ReadMessage()
		if e != nil {
			return
		}
		if mt != websocket.BinaryMessage || len(data) == 0 {
			continue
		}
		switch data[0] {
		case msgTTSOpus:
			payload := append([]byte(nil), data[1:]...)
			select {
			case s.out <- payload:
			default:
				<-s.out
				s.out <- payload
			}
		case msgClear:
			s.clearOutput()
		}
	}
}

func main() {
	listen := flag.String("listen", "127.0.0.1:8790", "HTTP/media IPC listen address")
	udpListen := flag.String("udp-listen", "0.0.0.0:8791", "private-network ICE UDP listen address")
	tcpListen := flag.String("tcp-listen", "0.0.0.0:8792", "private-network ICE TCP listen address")
	flag.Parse()
	udpAddr, err := net.ResolveUDPAddr("udp", *udpListen)
	if err != nil {
		log.Fatal(err)
	}
	udpConn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		log.Fatal(err)
	}
	setting := webrtc.SettingEngine{}
	setting.SetNetworkTypes([]webrtc.NetworkType{webrtc.NetworkTypeUDP4, webrtc.NetworkTypeTCP4})
	setting.SetICEUDPMux(ice.NewUDPMuxDefault(ice.UDPMuxParams{UDPConn: udpConn}))
	tcpAddr, err := net.ResolveTCPAddr("tcp", *tcpListen)
	if err != nil {
		log.Fatal(err)
	}
	tcpListener, err := net.ListenTCP("tcp", tcpAddr)
	if err != nil {
		log.Fatal(err)
	}
	setting.SetICETCPMux(webrtc.NewICETCPMux(nil, tcpListener, 8))
	m := &webrtc.MediaEngine{}
	if err = m.RegisterCodec(webrtc.RTPCodecParameters{
		RTPCodecCapability: webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2, SDPFmtpLine: "minptime=10;useinbandfec=1"},
		PayloadType:        111,
	}, webrtc.RTPCodecTypeAudio); err != nil {
		log.Fatal(err)
	}
	i := &interceptor.Registry{}
	if err = webrtc.RegisterDefaultInterceptors(m, i); err != nil {
		log.Fatal(err)
	}
	rtcAPI = webrtc.NewAPI(webrtc.WithMediaEngine(m), webrtc.WithInterceptorRegistry(i), webrtc.WithSettingEngine(setting))
	http.HandleFunc("/offer/", offerHandler)
	http.HandleFunc("/media/", mediaHandler)
	http.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { fmt.Fprintln(w, "ok") })
	log.Printf("VoiceMem RTC gateway HTTP=%s ICE/UDP=%s ICE/TCP=%s", *listen, *udpListen, *tcpListen)
	log.Fatal(http.ListenAndServe(*listen, nil))
}
