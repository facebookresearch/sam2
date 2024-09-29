#version 300 es
// Copyright (c) Meta Platforms, Inc. and affiliates.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

precision mediump float;

in vec2 vTexCoord;

uniform sampler2D uSampler;
uniform vec2 uSize;
uniform bool uSwapColor;
uniform bool uMonocolor;

out vec4 fragColor;

void main() {
  // calculate the offset for one pixel in texture coordinates
  vec2 texOffset = 1.0f / uSize;
  vec3 result = vec3(0.0f);
  // neighboring pixels
  vec3 tLeft = texture(uSampler, vTexCoord + texOffset * vec2(-1, 1)).rgb;
  vec3 tRight = texture(uSampler, vTexCoord + texOffset * vec2(1, -1)).rgb;
  vec3 bLeft = texture(uSampler, vTexCoord + texOffset * vec2(-1, -1)).rgb;
  vec3 bRight = texture(uSampler, vTexCoord + texOffset * vec2(1, 1)).rgb;
  
  // calculate the gradient edge of the current pixel using [3x3] sobel operator.
  vec3 xEdge = tLeft + 2.0f * texture(uSampler, vTexCoord + texOffset * vec2(-1, 0)).rgb + bLeft - tRight - 2.0f * texture(uSampler, vTexCoord + texOffset * vec2(1, 0)).rgb - bRight;
  vec3 yEdge = tLeft + 2.0f * texture(uSampler, vTexCoord + texOffset * vec2(0, 1)).rgb + tRight - bLeft - 2.0f * texture(uSampler, vTexCoord + texOffset * vec2(0, -1)).rgb - bRight;

  // magnitude of the gradient at the current pixel.
  result = sqrt(xEdge * xEdge + yEdge * yEdge);
  
  if (uMonocolor) {
    // Convert result to a grayscale intensity
    float intensity = length(result) / sqrt(3.0);
    // Threshold to determine if the color should be white or black
    float threshold = 0.2;
    if (intensity > threshold) {
      fragColor = uSwapColor ? vec4(1.0) : vec4(0.0, 0.0, 0.0, 1.0);
    } else {
      fragColor = uSwapColor ? vec4(0.0, 0.0, 0.0, 1.0) : vec4(1.0);
    }
  } else {
    result = uSwapColor ? result : vec3(0.0, 1.0, 0.0) * result;
    vec4 finalColor = vec4(result, 1.0f);
    fragColor = finalColor;
  }
}