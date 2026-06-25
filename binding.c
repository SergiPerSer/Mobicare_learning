#include "flappy_bird.h"
#define OBS_SIZE 160
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TENSOR_T ByteTensor

#define Env BirdEnv // O el nombre que le des a tu estructura en el wrapper de PufferLib
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->size = dict_get(kwargs, "size")->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
