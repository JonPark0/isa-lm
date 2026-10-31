/* Chat templates: the text of one conversation turn, as the models' own
 * templates (Hugging Face apply_chat_template) render it. */
#ifndef ML_CHAT_H
#define ML_CHAT_H

enum { ML_CHAT_LLAMA3 = 0, ML_CHAT_ZEPHYR = 1, ML_CHAT_LLAMA2 = 2, ML_CHAT_CHATML = 3, ML_CHAT_QWEN = 4,
       ML_CHAT_SMOLLM = 5 };

/* Template id for a name ("llama3", "zephyr", "llama2", "chatml", "qwen",
 * "smollm"), or -1. "qwen" and "smollm" are ChatML with Qwen2.5's and
 * SmolLM2's default system prompts. */
int ml_chat_template(const char *name);
/* The user turn `user` (the system prompt too on the first turn, if any)
 * followed by the assistant header; malloc'd, NULL if out of memory. */
char *ml_chat_render(int tpl, const char *system, const char *user, int first);
/* Whether each turn starts with BOS (llama2), or only the first one. */
int ml_chat_bos_each_turn(int tpl);
/* Text of the token that ends an assistant reply. */
const char *ml_chat_stop(int tpl);

#endif
