#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 out_color;

layout(set = 0, binding = 0) uniform ObjectUniforms {
	mat4 model;
	mat4 view;
	mat4 projection;
	vec4 color;
} u;

void main() {
	gl_Position = u.projection * u.view * u.model * vec4(in_position, 1.0);

	// Освещения ещё нет, поэтому нормали временно выводятся цветом,
	// чтобы проверить, что они сгенерированы верно
	vec3 normal = normalize(mat3(u.model) * in_normal);
	out_color = (normal * 0.5 + 0.5) * u.color.rgb;
}
