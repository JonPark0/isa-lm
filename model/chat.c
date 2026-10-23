#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model/chat.h"

int ml_chat_template(const char *name) {
    if (!strcmp(name, "llama3")) return ML_CHAT_LLAMA3;
    if (!strcmp(name, "zephyr")) return ML_CHAT_ZEPHYR;
    if (!strcmp(name, "llama2")) return ML_CHAT_LLAMA2;
    return -1;
}

int ml_chat_bos_each_turn(int tpl) { return tpl == ML_CHAT_LLAMA2; }

const char *ml_chat_stop(int tpl) { return tpl == ML_CHAT_LLAMA3 ? "<|eot_id|>" : "</s>"; }

/* Each turn starts right after a special token (BOS, or the previous reply's
 * stop token), so encoding turn by turn gives the ids of the whole rendered
 * conversation. */
char *ml_chat_render(int tpl, const char *system, const char *user, int first) {
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
    default:
        free(out);
        return NULL;
    }
    return out;
}
