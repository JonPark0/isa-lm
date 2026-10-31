#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model/chat.h"

int ml_chat_template(const char *name) {
    if (!strcmp(name, "llama3")) return ML_CHAT_LLAMA3;
    if (!strcmp(name, "zephyr")) return ML_CHAT_ZEPHYR;
    if (!strcmp(name, "llama2")) return ML_CHAT_LLAMA2;
    if (!strcmp(name, "chatml")) return ML_CHAT_CHATML;
    if (!strcmp(name, "qwen")) return ML_CHAT_QWEN;
    if (!strcmp(name, "smollm")) return ML_CHAT_SMOLLM;
    if (!strcmp(name, "gemma")) return ML_CHAT_GEMMA;
    return -1;
}

int ml_chat_bos_each_turn(int tpl) { return tpl == ML_CHAT_LLAMA2; }

const char *ml_chat_stop(int tpl) {
    if (tpl == ML_CHAT_LLAMA3) return "<|eot_id|>";
    if (tpl == ML_CHAT_GEMMA) return "<end_of_turn>";
    return tpl == ML_CHAT_CHATML || tpl == ML_CHAT_QWEN || tpl == ML_CHAT_SMOLLM ? "<|im_end|>" : "</s>";
}

static const char QWEN_SYSTEM[] = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
static const char SMOLLM_SYSTEM[] = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

/* Each turn starts right after a special token (BOS, or the previous reply's
 * stop token), so encoding turn by turn gives the ids of the whole rendered
 * conversation. */
char *ml_chat_render(int tpl, const char *system, const char *user, int first) {
    if (tpl == ML_CHAT_QWEN && !(system && system[0])) system = QWEN_SYSTEM;
    if (tpl == ML_CHAT_SMOLLM && !(system && system[0])) system = SMOLLM_SYSTEM;
    const int sys = first && system && system[0];
    const size_t n = strlen(user) + (sys ? strlen(system) : 0) + 256;   /* template text is under 200 bytes */
    char *out = malloc(n);
    if (!out) return NULL;
    out[0] = '\0';
    switch (tpl) {
    case ML_CHAT_LLAMA3:
        if (sys) snprintf(out, n, "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>", system);
        snprintf(out + strlen(out), n - strlen(out),
                 "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", user);
        break;
    case ML_CHAT_ZEPHYR:   /* TinyLlama chat: a reply is "...</s>\n" */
        if (!first) strcpy(out, "\n");
        if (sys) snprintf(out + strlen(out), n - strlen(out), "<|system|>\n%s</s>\n", system);
        snprintf(out + strlen(out), n - strlen(out), "<|user|>\n%s</s>\n<|assistant|>\n", user);
        break;
    case ML_CHAT_LLAMA2:   /* also Mistral Instruct, without a system prompt */
        if (sys) snprintf(out, n, "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]", system, user);
        else snprintf(out, n, "[INST] %s [/INST]", user);
        break;
    case ML_CHAT_CHATML:   /* also Qwen2.5: a reply is "...<|im_end|>\n" */
    case ML_CHAT_QWEN:
    case ML_CHAT_SMOLLM:
        if (!first) strcpy(out, "\n");
        if (sys) snprintf(out + strlen(out), n - strlen(out), "<|im_start|>system\n%s<|im_end|>\n", system);
        snprintf(out + strlen(out), n - strlen(out), "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user);
        break;
    case ML_CHAT_GEMMA:   /* a reply is "...<end_of_turn>\n"; no system role */
        if (!first) strcpy(out, "\n");
        snprintf(out + strlen(out), n - strlen(out), "<start_of_turn>user\n%s%s%s<end_of_turn>\n<start_of_turn>model\n",
                 sys ? system : "", sys ? "\n\n" : "", user);
        break;
    default:
        free(out);
        return NULL;
    }
    return out;
}
