#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 out_color;

layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
} scene;

layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	vec4 color;
} model;

void main() {
	gl_Position = scene.view_projection * model.model * vec4(in_position, 1.0);

	// Освещения ещё нет, поэтому нормали временно выводятся цветом,
	// чтобы проверить, что они сгенерированы верно
	vec3 normal = normalize(mat3(model.model) * in_normal);
	out_color = (normal * 0.5 + 0.5) * model.color.rgb;
}
