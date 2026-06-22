#include "flappy_bird.h"
#include "puffernet.h"

void demo() {
    BirdEnv env = {.size = 32}; // 32 Columnas
    int total_tiles = MAP_ROWS * env.size; // 5 * 32 = 160

    env.observation = (unsigned char*)calloc(total_tiles, sizeof(unsigned char));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));
    
    // Corregida la asignación de memoria simulada de pesos para que no falle al arrancar
    Weights weights = {.size = 1000, .weights = (float*)calloc(1000, sizeof(float))};
    int logit_sizes[1] = {5};
    
    // Corregido: El tercer parámetro debe coincidir exactamente con el total de observaciones (160)
    PufferNet* net = make_puffernet(weights, 1, total_tiles, 128, 1, logit_sizes, 1);
    
    duck_hunt(&env);
    c_render(&env);
    
    while (!WindowShouldClose()) {
        // Copiar las observaciones al formato float que requiere PufferNet
        float obs_f[160]; 
        for (int i = 0; i < total_tiles; i++) {
            obs_f[i] = (float)env.observation[i];
        }
        
        // Decisión de la IA
        forward_puffernet(net, obs_f, env.actions);
        
        c_step(&env);
        c_render(&env);
    }

    // Liberación segura de memoria sin leaks
    free_puffernet(net);
    free(weights.weights);
    free(env.observation);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

int main() {
    demo();
    return 0;
}