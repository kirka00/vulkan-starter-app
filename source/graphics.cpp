#include "graphics.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>

#include <vulkan/vulkan.h>

#include "graphics_internal.hpp"

namespace graphics {

bool createBuffer(Buffer& out, VkDeviceSize size, VkBufferUsageFlags usage,
                  const void* initial_data) {
	auto& context = internal::context;

	const VkBufferCreateInfo buffer = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo allocation_info = {};

	if (vmaCreateBuffer(context.allocator, &buffer, &allocation, &out.buffer,
	                    &out.allocation, &allocation_info) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer of " << size << " bytes\n";
		return false;
	}

	out.mapped = allocation_info.pMappedData;
	out.size = size;

	if (initial_data != nullptr) {
		uploadBuffer(out, initial_data, size);
	}

	return true;
}

bool createDeviceBuffer(Buffer& out, VkDeviceSize size, VkBufferUsageFlags usage) {
	const VkBufferCreateInfo buffer = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	// Без флагов доступа с CPU VMA выбирает DEVICE_LOCAL память
	const VmaAllocationCreateInfo allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateBuffer(internal::context.allocator, &buffer, &allocation, &out.buffer,
	                    &out.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create device-local buffer of " << size << " bytes\n";
		return false;
	}

	out.mapped = nullptr;
	out.size = size;
	return true;
}

void uploadBuffer(Buffer& buffer, const void* data, VkDeviceSize size, VkDeviceSize offset) {
	std::memcpy(static_cast<uint8_t*>(buffer.mapped) + offset, data, size_t(size));
	// На host-coherent памяти ничего не делает, на остальной — обязателен
	vmaFlushAllocation(internal::context.allocator, buffer.allocation, offset, size);
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(internal::context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

namespace {

// Смещение копирования из буфера в изображение должно быть кратно размеру texel'а (4 байта),
// 16 берём с запасом
constexpr VkDeviceSize staging_alignment = 16;

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
	return (value + alignment - 1) & ~(alignment - 1);
}

// Кладёт данные в staging-буфер и возвращает их смещение, или UINT64_MAX, если места нет
VkDeviceSize stage(Uploader& up, const void* data, VkDeviceSize size) {
	const VkDeviceSize offset = alignUp(up.offset, staging_alignment);
	if (offset + size > up.staging.size) {
		std::cerr << "Staging buffer is too small: need " << offset + size
		          << " bytes, have " << up.staging.size << '\n';
		return UINT64_MAX;
	}
	uploadBuffer(up.staging, data, size, offset);
	up.offset = offset + size;
	return offset;
}

void imageBarrier(VkCommandBuffer cmd, VkImage image, uint32_t base_mip, uint32_t mip_count,
                  VkImageLayout old_layout, VkImageLayout new_layout,
                  VkAccessFlags src_access, VkAccessFlags dst_access,
                  VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
	const VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = src_access,
		.dstAccessMask = dst_access,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = base_mip,
			.levelCount = mip_count,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

} // namespace

VkDeviceSize uploadFootprint(VkDeviceSize size) {
	return alignUp(size, staging_alignment);
}

bool beginUpload(Uploader& up, VkDeviceSize capacity) {
	auto& context = internal::context;

	if (!createBuffer(up.staging, capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
		return false;
	}
	up.offset = 0;

	const VkCommandPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};

	if (vkCreateCommandPool(context.device, &pool, nullptr, &up.command_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create the upload command pool\n";
		return false;
	}

	const VkCommandBufferAllocateInfo allocate = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = up.command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	if (vkAllocateCommandBuffers(context.device, &allocate, &up.command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to allocate the upload command buffer\n";
		return false;
	}

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	if (vkBeginCommandBuffer(up.command_buffer, &begin) != VK_SUCCESS) {
		std::cerr << "Failed to begin recording of the upload command buffer\n";
		return false;
	}

	return true;
}

bool uploadToBuffer(Uploader& up, const Buffer& dst, const void* data, VkDeviceSize size) {
	const VkDeviceSize offset = stage(up, data, size);
	if (offset == UINT64_MAX) {
		return false;
	}

	const VkBufferCopy copy = {
		.srcOffset = offset,
		.dstOffset = 0,
		.size = size,
	};
	vkCmdCopyBuffer(up.command_buffer, up.staging.buffer, dst.buffer, 1, &copy);
	return true;
}

bool uploadToTexture(Uploader& up, Texture& out, const void* pixels,
                     uint32_t width, uint32_t height, VkFormat format) {
	auto& context = internal::context;

	// Мип-уровни строятся на GPU через vkCmdBlitImage с линейной фильтрацией,
	// формат должен это поддерживать, иначе обходимся одним уровнем
	VkFormatProperties format_props = {};
	vkGetPhysicalDeviceFormatProperties(context.physical_device, format, &format_props);
	const bool can_blit = format_props.optimalTilingFeatures &
	                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

	uint32_t mip_levels = 1;
	if (can_blit) {
		for (uint32_t size = std::max(width, height); size > 1; size /= 2) {
			++mip_levels;
		}
	}

	const VkImageCreateInfo image = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { width, height, 1 },
		.mipLevels = mip_levels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
		         VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateImage(context.allocator, &image, &allocation, &out.image,
	                   &out.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create a " << width << "x" << height << " image\n";
		return false;
	}

	out.width = width;
	out.height = height;
	out.mip_levels = mip_levels;

	const VkDeviceSize size = VkDeviceSize(width) * height * 4;
	const VkDeviceSize offset = stage(up, pixels, size);
	if (offset == UINT64_MAX) {
		return false;
	}

	VkCommandBuffer cmd = up.command_buffer;

	// Все уровни из неопределённой раскладки в приёмник копирования
	imageBarrier(cmd, out.image, 0, mip_levels,
	             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	             0, VK_ACCESS_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	const VkBufferImageCopy copy = {
		.bufferOffset = offset,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,
		.imageSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.imageOffset = { 0, 0, 0 },
		.imageExtent = { width, height, 1 },
	};
	vkCmdCopyBufferToImage(cmd, up.staging.buffer, out.image,
	                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

	// Каждый следующий уровень — уменьшенная вдвое копия предыдущего
	int32_t mip_width = int32_t(width);
	int32_t mip_height = int32_t(height);
	for (uint32_t i = 1; i < mip_levels; ++i) {
		imageBarrier(cmd, out.image, i - 1, 1,
		             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		             VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
		             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

		const int32_t next_width = std::max(mip_width / 2, 1);
		const int32_t next_height = std::max(mip_height / 2, 1);

		const VkImageBlit blit = {
			.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 },
			.srcOffsets = { { 0, 0, 0 }, { mip_width, mip_height, 1 } },
			.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 },
			.dstOffsets = { { 0, 0, 0 }, { next_width, next_height, 1 } },
		};
		vkCmdBlitImage(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		               out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		               1, &blit, VK_FILTER_LINEAR);

		// Уровень i - 1 больше не меняется, отдаём его фрагментному шейдеру
		imageBarrier(cmd, out.image, i - 1, 1,
		             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		             VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
		             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		mip_width = next_width;
		mip_height = next_height;
	}

	// Последний уровень ни разу не был источником, переводим его отдельно
	imageBarrier(cmd, out.image, mip_levels - 1, 1,
	             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	             VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
	             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

	const VkImageViewCreateInfo view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = out.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = mip_levels,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &view, nullptr, &out.view) != VK_SUCCESS) {
		std::cerr << "Failed to create an image view\n";
		return false;
	}

	return true;
}

bool endUpload(Uploader& up) {
	auto& context = internal::context;

	// Результат копирований в вершинные и индексные буферы должен стать видимым
	// для этапа чтения вершин в следующих кадрах
	const VkMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
	};
	vkCmdPipelineBarrier(up.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

	vkEndCommandBuffer(up.command_buffer);

	const VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &up.command_buffer,
	};

	// Загрузка разовая, при старте, поэтому достаточно дождаться простоя очереди
	const bool ok = vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
	                vkQueueWaitIdle(context.graphics_queue) == VK_SUCCESS;
	if (!ok) {
		std::cerr << "Failed to submit the upload command buffer\n";
	}

	vkDestroyCommandPool(context.device, up.command_pool, nullptr);
	destroyBuffer(up.staging);
	up = {};
	return ok;
}

void destroyTexture(Texture& texture) {
	auto& context = internal::context;
	if (texture.view != VK_NULL_HANDLE) {
		vkDestroyImageView(context.device, texture.view, nullptr);
	}
	if (texture.image != VK_NULL_HANDLE) {
		vmaDestroyImage(context.allocator, texture.image, texture.allocation);
	}
	texture = {};
}

void destroyMesh(Mesh& mesh) {
	destroyBuffer(mesh.index_buffer);
	destroyBuffer(mesh.vertex_buffer);
	mesh = {};
}

VkShaderModule loadShaderModule(const std::string& path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file " << path
		          << " (is the working directory the project root?)\n";
		return VK_NULL_HANDLE;
	}

	const std::streamsize size = file.tellg();
	file.seekg(0);

	std::vector<uint32_t> code((size_t(size) + sizeof(uint32_t) - 1) / sizeof(uint32_t));
	file.read(reinterpret_cast<char*>(code.data()), size);

	const VkShaderModuleCreateInfo module = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size_t(size),
		.pCode = code.data(),
	};

	VkShaderModule result = VK_NULL_HANDLE;
	if (vkCreateShaderModule(internal::context.device, &module, nullptr, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module from " << path << '\n';
		return VK_NULL_HANDLE;
	}

	return result;
}

VkPipeline createGraphicsPipeline(const GraphicsPipelineDesc& desc) {
	auto& context = internal::context;

	VkShaderModule vertex_shader = loadShaderModule(desc.vertex_shader_path);
	VkShaderModule fragment_shader = loadShaderModule(desc.fragment_shader_path);
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return VK_NULL_HANDLE;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = uint32_t(desc.vertex_bindings.size()),
		.pVertexBindingDescriptions = desc.vertex_bindings.data(),
		.vertexAttributeDescriptionCount = uint32_t(desc.vertex_attributes.size()),
		.pVertexAttributeDescriptions = desc.vertex_attributes.data(),
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = desc.topology,
	};

	// Сами viewport и scissor задаются через vkCmdSetViewport и vkCmdSetScissor
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = desc.polygon_mode,
		.cullMode = desc.cull_mode,
		.frontFace = desc.front_face,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = desc.depth_test,
		.depthWriteEnable = desc.depth_write,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = desc.layout,
		.renderPass = desc.render_pass,
		.subpass = 0,
	};

	VkPipeline result = VK_NULL_HANDLE;
	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline,
	                              nullptr, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		result = VK_NULL_HANDLE;
	}

	// Модули уже встроены в конвейер, их можно сразу освободить
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	return result;
}

} // namespace graphics
