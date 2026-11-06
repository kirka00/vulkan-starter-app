#version 450

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

struct PointLight {
	vec4 position;  // xyz = позиция
	vec4 color;     // rgb = цвет, a = интенсивность
};

struct SpotLight {
	vec4 position;   // xyz = позиция, w = косинус внутреннего угла
	vec4 direction;  // xyz = направление, w = косинус внешнего угла
	vec4 color;      // rgb = цвет, a = интенсивность
};

layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 sun_direction;
	vec4 sun_color;
	uvec4 light_counts;
} scene;

layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer {
	PointLight lights[];
} point_lights;

layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer {
	SpotLight lights[];
} spot_lights;

layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec4 specular;
	vec4 emissive;
} model;

// Набор материала: у каждого материала свои три текстуры, у отсутствующих —
// служебная текстура из одного чёрного пикселя
layout(set = 1, binding = 1) uniform sampler2D albedo_map;
layout(set = 1, binding = 2) uniform sampler2D specular_map;
layout(set = 1, binding = 3) uniform sampler2D emissive_map;

// Флаги меняются от объекта к объекту командой vkCmdPushConstants,
// в uniform-буфер модели их не кладём
layout(push_constant) uniform PushConstants {
	uint flags;  // бит 0 — нетривиальное сэмплирование
	float time;  // секунды с запуска
} push;

const uint FLAG_ANIMATED = 1u;

// Одна и та же текстура читается дважды по разным координатам, которые меняются со временем
vec3 sampleAnimated(sampler2D map, vec2 uv) {
	float t = push.time;
	vec2 wave = 0.015 * vec2(sin(uv.y * 30.0 + t * 2.0), cos(uv.x * 30.0 + t * 1.6));
	vec3 a = texture(map, uv + wave + vec2(0.04 * t, 0.0)).rgb;
	vec3 b = texture(map, uv * 1.7 - wave + vec2(-0.025 * t, 0.05 * t)).rgb;
	return mix(a, b, 0.5);
}

vec3 blinnPhong(vec3 n, vec3 v, vec3 l, vec3 radiance, vec3 albedo, vec3 specular_color) {
	float n_dot_l = max(dot(n, l), 0.0);

	vec3 diffuse = albedo * n_dot_l;

	vec3 h = normalize(l + v);
	float highlight = n_dot_l > 0.0 ? pow(max(dot(n, h), 0.0), max(model.specular.a, 1.0)) : 0.0;
	vec3 specular = specular_color * highlight;

	return (diffuse + specular) * radiance;
}

void main() {
	vec3 n = normalize(in_normal);
	vec3 v = normalize(scene.camera_position.xyz - in_world_position);

	vec3 albedo;
	vec3 emission;
	if ((push.flags & FLAG_ANIMATED) != 0u) {
		float t = push.time;
		// Модуляция: цвет изображения медленно переливается вдоль высоты объекта
		vec3 tint = 0.8 + 0.2 * cos(t * 1.5 + in_uv.y * 6.2831 + vec3(0.0, 2.1, 4.2));
		albedo = sampleAnimated(albedo_map, in_uv) * tint;
		emission = sampleAnimated(emissive_map, in_uv) * (0.7 + 0.3 * sin(t * 3.0));
	} else {
		albedo = texture(albedo_map, in_uv).rgb;
		emission = texture(emissive_map, in_uv).rgb;
	}

	// Текстура говорит, где блестеть, uniform — каким цветом и насколько сильно
	vec3 specular_color = model.specular.rgb * texture(specular_map, in_uv).rgb;

	vec3 color = scene.ambient.rgb * scene.ambient.a * albedo;

	color += blinnPhong(n, v, normalize(-scene.sun_direction.xyz),
	                    scene.sun_color.rgb * scene.sun_color.a, albedo, specular_color);

	for (uint i = 0; i < scene.light_counts.x; ++i) {
		PointLight light = point_lights.lights[i];

		vec3 to_light = light.position.xyz - in_world_position;
		float distance = max(length(to_light), 0.001);
		float falloff = 1.0 / (distance * distance);

		color += blinnPhong(n, v, to_light / distance, light.color.rgb * light.color.a * falloff,
		                    albedo, specular_color);
	}

	for (uint i = 0; i < scene.light_counts.y; ++i) {
		SpotLight light = spot_lights.lights[i];

		vec3 to_light = light.position.xyz - in_world_position;
		float distance = max(length(to_light), 0.001);
		vec3 l = to_light / distance;
		float falloff = 1.0 / (distance * distance);

		float cosine = dot(normalize(light.direction.xyz), -l);
		float cone = smoothstep(light.direction.w, light.position.w, cosine);

		color += blinnPhong(n, v, l, light.color.rgb * light.color.a * falloff * cone,
		                    albedo, specular_color);
	}

	// Свечение прибавляется после освещения и от источников не зависит
	color += emission * model.emissive.rgb * model.emissive.a;

	out_color = vec4(color, 1.0);
}
