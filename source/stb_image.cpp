// Реализация stb_image собирается в отдельной единице трансляции, чтобы её
// предупреждения не смешивались с кодом лабораторной
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include <stb_image.h>
