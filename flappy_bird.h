#ifndef FLAPPY_BIRD_H
#define FLAPPY_BIRD_H

#include <stdlib.h>
#include <string.h>
#include "raylib.h"

// Acciones del agente
const unsigned char NOOP = 0;
const unsigned char FLAP_UP = 1;
const unsigned char FLAP_DOWN = 2;
const unsigned char FLAP_LEFT = 3;
const unsigned char FLAP_RIGHT = 4;

// Entidades del mapa
const unsigned char EMPTY = 0;
const unsigned char PIPE = 1;
const unsigned char BIRD = 2;
const unsigned char GOAL = 3;

// Constante fija para las filas del mapa de Flappy
#define MAP_ROWS 5

typedef struct {
    float perf; 
    float score; 
    float episode_return; 
    float episode_length; 
    float n;
} Log;

typedef struct {
    Log log; 
    unsigned char* observation; 
    float* actions; 
    float* rewards; 
    float* terminals; 
    int size; // Representa las Columnas (cols)
    int tick;
    int x;
    int y;
    unsigned int rng;
    int num_agents;
} BirdEnv;

// Función corregida para actualizar estadísticas de entrenamiento
void log_data(BirdEnv* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1.0f : 0.0f;
    env->log.score += env->rewards[0];
    env->log.episode_return += env->rewards[0]; // Corregido
    env->log.episode_length += (float)env->tick; // Corregido
    env->log.n++;
}

// Inicialización/Reinicio del mapa (Duck Hunt / Flappy Bird)
void duck_hunt(BirdEnv* env) { 
    int cols = env->size;
    int tiles = MAP_ROWS * cols; 
    memset(env->observation, 0, tiles * sizeof(unsigned char)); 
    
    int center_row = MAP_ROWS / 2;
    env->x = 0;
    env->y = center_row;
    env->observation[center_row * cols] = BIRD; 
    env->tick = 0;

    int pipe_count = cols / 4; 
    if (pipe_count < 1) pipe_count = 1; 
    int spacing = 2; 
    int start_col = 2; 

    unsigned int state = env->rng; 
    if (state == 0) state = 1; 

    for (int i = 0; i < pipe_count; i++) { 
        int col = start_col + i * (1 + spacing); 
        if (col >= cols) break; 
        
        state = state * 1664525u + 1013904223u; 
        int hole_row = state % MAP_ROWS; 
        
        for (int r = 0; r < MAP_ROWS; r++) { 
            env->observation[r * cols + col] = (r == hole_row) ? GOAL : PIPE; 
        }
    }
    env->rng = state;
}

// Avanzar un paso en la simulación
void c_step(BirdEnv* env) {
    env->tick++; 
    int action = (int)env->actions[0]; 
    env->terminals[0] = 0; 
    env->rewards[0] = 0; 
    
    // Borrar posición anterior usando las columnas correctas (env->size)
    env->observation[env->y * env->size + env->x] = EMPTY;

    if (action == FLAP_DOWN)   env->y++;
    else if (action == FLAP_UP)     env->y--;
    else if (action == FLAP_LEFT)   env->x--;
    else if (action == FLAP_RIGHT)  env->x++;

    // VALIDACIÓN CRÍTICA DE LÍMITES CORREGIDA (Eje Y limitado por MAP_ROWS)
    if (env->x < 0 || env->x >= env->size || env->y < 0 || env->y >= MAP_ROWS) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0f;
        log_data(env); // Corregido nombre de función
        duck_hunt(env); 
        return; 
    }

    int pos = env->y * env->size + env->x; 

    // Colisión con un tubo
    if (env->observation[pos] == PIPE) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0f;
        log_data(env);
        duck_hunt(env);
        return;
    }

    // Pasar por la zona segura (Agujero / GOAL)
    if (env->observation[pos] == GOAL) {
        env->rewards[0] = 1.0f;
        // No reiniciamos el juego aquí, dejamos que continúe acumulando puntos
    }

    // Llegar al final de la pantalla (Victoria)
    if (env->x == env->size - 1) {
        env->terminals[0] = 1;
        env->rewards[0] = 2.0f; // Recompensa extra por ganar
        log_data(env); 
        duck_hunt(env); 
        return;
    }   

    // Actualizar nueva posición del pájaro
    env->observation[pos] = BIRD; 
}

void c_render(BirdEnv* env) {
    int cols = env->size;

    if (!IsWindowReady()) {
        InitWindow(64 * cols, 64 * MAP_ROWS, "PufferLib Flappy Bird");
        SetTargetFPS(5);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int px = 64;
    for (int i = 0; i < MAP_ROWS; i++) {
        for (int j = 0; j < cols; j++) {
            int tex = env->observation[i * cols + j];
            if (tex == EMPTY) continue;
            
            Color color = (Color){255, 255, 255, 255}; // Blanco por defecto
            if (tex == PIPE)        color = (Color){187, 0, 0, 255};    // Rojo
            else if (tex == GOAL)   color = (Color){0, 187, 0, 255};    // Verde
            else if (tex == BIRD)   color = (Color){0, 0, 187, 255};    // Azul
            
            DrawRectangle(j * px, i * px, px, px, color);
        }
    }
    EndDrawing();
}

void c_close(BirdEnv* env) { // Corregido el tipo de parámetro
    if (IsWindowReady()) {
        CloseWindow();
    }
}

#endif