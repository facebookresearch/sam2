/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
export async function handleSaveVideo(
  videoPath: string,
  fileName?: string,
): Promise<void> {
  const blob = await fetch(videoPath).then(res => res.blob());

  return new Promise(resolve => {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.addEventListener('load', () => {
      const elem = document.createElement('a');
      elem.download = fileName ?? getFileName();
      if (typeof reader.result === 'string') {
        elem.href = reader.result;
      }
      elem.click();
      resolve();
    });
  });
}

export function getFileName() {
  const date = new Date();
  const timestamp = date.getTime();
  return `sam2_masked_video_${timestamp}.mp4`;
}
