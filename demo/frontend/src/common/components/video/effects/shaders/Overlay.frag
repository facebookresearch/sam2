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

precision highp float;

in vec2 vTexCoord;

uniform sampler2D uSampler;
uniform vec2 uSize;
uniform int uNumMasks;
uniform float uOpacity;
uniform bool uBorder;
uniform sampler2D uMaskTexture0;
uniform sampler2D uMaskTexture1;
uniform sampler2D uMaskTexture2;

uniform vec4 uMaskColor0;
uniform vec4 uMaskColor1;
uniform vec4 uMaskColor2;

uniform float uTime;
uniform vec2 uClickPos;
uniform int uActiveMask;

out vec4 fragColor;

vec4 lowerSaturation(vec4 color, float saturationFactor) {
  float luminance = 0.299f * color.r + 0.587f * color.g + 0.114f * color.b; // Calculate luminance
  vec3 gray = vec3(luminance);
  vec3 saturated = mix(gray, color.rgb, saturationFactor); // Mix gray with original color based on saturation factor
  return vec4(saturated, color.a);
}

vec4 detectEdges(sampler2D textureSampler, float coverage, vec4 edgeColor) {
  vec2 tvTexCoord = vec2(vTexCoord.y, vTexCoord.x);
  vec2 texOffset = 1.0f / uSize;
  vec3 result = vec3(0.0f);
  // neighboring pixels
  vec3 tLeft = texture(textureSampler, tvTexCoord + texOffset * vec2(-coverage, coverage)).rgb;
  vec3 tRight = texture(textureSampler, tvTexCoord + texOffset * vec2(coverage, -coverage)).rgb;
  vec3 bLeft = texture(textureSampler, tvTexCoord + texOffset * vec2(-coverage, -coverage)).rgb;
  vec3 bRight = texture(textureSampler, tvTexCoord + texOffset * vec2(coverage, coverage)).rgb;

  // calculate the gradient edge of the current pixel using [3x3] sobel operator.
  vec3 xEdge = tLeft + 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(-coverage, 0)).rgb + bLeft - tRight - 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(coverage, 0)).rgb - bRight;
  vec3 yEdge = tLeft + 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(0, coverage)).rgb + tRight - bLeft - 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(0, -coverage)).rgb - bRight;

  // magnitude of the gradient at the current pixel.
  result = sqrt(xEdge * xEdge + yEdge * yEdge);
  return result.r > 1e-6f ? edgeColor : vec4(0.0f, 0.0f, 0.0f, 0.0f);
}

vec2 calculateAdjustedTexCoord(vec2 vTexCoord, vec4 bbox, float aspectRatio) {
  vec2 center = vec2((bbox.x + bbox.z) * 0.5f, bbox.w);
  float radiusX = abs(bbox.z - bbox.x);
  float radiusY = radiusX / aspectRatio;
  float scale = 1.0f;
  radiusX *= scale;
  radiusY *= scale;
  vec2 adjustedTexCoord = (vTexCoord - center) / vec2(radiusX, radiusY) + vec2(0.5f);
  return adjustedTexCoord;
}

void main() {
  vec4 color = texture(uSampler, vTexCoord);
  vec4 color1 = uMaskColor0 / 255.0;
  vec4 color2 = uMaskColor1 / 255.0;
  vec4 color3 = uMaskColor2 / 255.0;
  float saturationFactor = 0.7;
  float aspectRatio = uSize.y / uSize.x;
  vec2 tvTexCoord = vec2(vTexCoord.y, vTexCoord.x);

  vec4 finalColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  float totalMaskValue = 0.0f;
  vec4 edgeColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  float numRipples = 1.75;
  float timeThreshold = 1.1; // can take any value from [0.0, 1.5]
  vec2 adjustedClickCoord =  calculateAdjustedTexCoord(vTexCoord, vec4(uClickPos, uClickPos + 0.1), aspectRatio);

  if(uNumMasks > 0) {
    float maskValue0 = texture(uMaskTexture0, tvTexCoord).r;
    vec4 saturatedColor = lowerSaturation(color1, saturationFactor);
    vec4 plainColor= vec4(vec3(saturatedColor).rgb, 1.0);
    vec4 rippleColor = vec4(color1.rgb, 0.2);
    
    if (uActiveMask == 0 && uTime < timeThreshold) {
      float dist = length(adjustedClickCoord);
      float colorFactor = abs(sin((dist - uTime) * numRipples));
      plainColor = vec4(mix(rippleColor, plainColor, colorFactor));
    };
    
    if (uTime >= timeThreshold) {
      plainColor= vec4(vec3(saturatedColor).rgb, 1.0);
    }
    finalColor += maskValue0 * plainColor;
    totalMaskValue += maskValue0;

    edgeColor = detectEdges(uMaskTexture0, 1.25, color1);
  }
  if(uNumMasks > 1) {
    float maskValue1 = texture(uMaskTexture1, tvTexCoord).r;
    vec4 saturatedColor = lowerSaturation(color2, saturationFactor);
    vec4 plainColor= vec4(vec3(saturatedColor).rgb, 1.0);
    vec4 rippleColor = vec4(color2.rgb, 0.2);

    if (uActiveMask == 1 && uTime < timeThreshold) {
      float dist = length(adjustedClickCoord);
      float colorFactor = abs(sin((dist - uTime) * numRipples));
      plainColor = vec4(mix(rippleColor, plainColor, colorFactor));
    }

    if (uTime >= timeThreshold) {
      plainColor= vec4(vec3(saturatedColor).rgb, 1.0);
    }
    finalColor += maskValue1 * plainColor;
    totalMaskValue += maskValue1;

    if(edgeColor.a <= 0.0f) {
      edgeColor = detectEdges(uMaskTexture1, 1.25, color2);
    }
  }
  if(uNumMasks > 2) {
    float maskValue2 = texture(uMaskTexture2, tvTexCoord).r;
    vec4 saturatedColor = lowerSaturation(color3, saturationFactor);
    vec4 plainColor= vec4(vec3(saturatedColor).rgb, 1.0);
    vec4 rippleColor = vec4(color3.rgb, 0.2);

    if (uActiveMask == 2 && uTime < timeThreshold) {
      float dist = length(adjustedClickCoord);
      float colorFactor = abs(sin((dist - uTime) * numRipples));
      plainColor = vec4(mix(rippleColor, plainColor, colorFactor));
    }

    if (uTime >= timeThreshold) {
      plainColor= vec4(vec3(saturatedColor).rgb, 1.0);
    }

    finalColor += maskValue2 * plainColor;
    totalMaskValue += maskValue2;

    if(edgeColor.a <= 0.0f) {
      edgeColor = detectEdges(uMaskTexture2, 1.25, color3);
    }
  }

  if(totalMaskValue > 0.0f) {
    finalColor /= totalMaskValue;
    finalColor = mix(color, finalColor, uOpacity);
  } else {
    finalColor.a = 0.0f;
  }

  if(edgeColor.a > 0.0f && uBorder) {
    finalColor = vec4(vec3(edgeColor), 1.0f);
  }
  fragColor = finalColor;
}