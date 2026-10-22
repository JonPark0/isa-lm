#include "core/ml.h"

const char *ml_status_str(ml_status s) {
    switch (s) {
    case ML_OK: return "ok";
    case ML_E_IO: return "i/o error";
    case ML_E_FORMAT: return "bad file format";
    case ML_E_SHAPE: return "bad tensor shape";
    case ML_E_NOMEM: return "out of memory";
    case ML_E_ISA: return "CPU lacks required ISA";
    }
    return "unknown";
}
