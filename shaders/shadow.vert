#version 450

layout(location = 0) in vec3 in_position;

layout(push_constant) uniform PushConstants {
	mat4 light_mvp;
} push;

void main() {
	gl_Position = push.light_mvp * vec4(in_position, 1.0);
}
