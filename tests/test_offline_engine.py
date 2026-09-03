import unittest

from voicemem.lang import set_memory_language
from voicemem.rightbrain.brain import RightBrain


class _RecordingTraitStore:
    def __init__(self):
        self.added = []

    def add(self, *args):
        self.added.append(args)
        return True


class ReactionTraitLanguageTest(unittest.TestCase):
    def tearDown(self):
        set_memory_language(None)

    def test_english_memory_rejects_chinese_reaction_trait(self):
        set_memory_language("en")
        store = _RecordingTraitStore()
        brain = object.__new__(RightBrain)
        brain._user_id = "user"
        brain._traits = lambda: store
        brain._attribute_reaction = lambda *_: {
            "significant": True,
            "assistant_helped": False,
            "user_trait": {
                "slot": "表达风格",
                "label": "不满时直说不绕弯",
            },
        }

        brain.learn_from_reaction(
            "No, that is not what I meant.",
            "annoyed",
            [],
            "Let me give you a plan.",
        )

        self.assertEqual(store.added, [])

    def test_english_memory_stores_english_reaction_trait(self):
        set_memory_language("en")
        store = _RecordingTraitStore()
        brain = object.__new__(RightBrain)
        brain._user_id = "user"
        brain._traits = lambda: store
        brain._attribute_reaction = lambda *_: {
            "significant": True,
            "assistant_helped": False,
            "user_trait": {
                "slot": "表达风格",
                "label": "speaks plainly when dissatisfied",
            },
        }

        brain.learn_from_reaction(
            "No, that is not what I meant.",
            "annoyed",
            [],
            "Let me give you a plan.",
        )

        self.assertEqual(len(store.added), 1)
        self.assertEqual(store.added[0][2], "speaks plainly when dissatisfied")

    def test_english_memory_uses_english_reaction_prompt(self):
        set_memory_language("en")
        prompts = []
        brain = object.__new__(RightBrain)
        brain._llm_json = lambda prompt: prompts.append(prompt) or '{"significant": false}'

        brain._attribute_reaction(
            "No, that is not what I meant.",
            "Let me give you a plan.",
            "annoyed",
        )

        self.assertIn("The assistant's reply:", prompts[0])
        self.assertIn("shuts down when given solutions", prompts[0])
        self.assertNotIn("被直接给方案会关闭", prompts[0])


if __name__ == "__main__":
    unittest.main()
