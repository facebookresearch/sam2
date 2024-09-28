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
type Props = {
  thumbnail: string | null;
  color: string;
  onClick?: () => void;
};

export default function ObjectThumbnail({thumbnail, color, onClick}: Props) {
  return (
    <div
      className="relative h-12 w-12 md:w-20 md:h-20 shrink-0 p-2 rounded-lg bg-contain bg-no-repeat bg-center"
      style={{
        backgroundColor: color,
      }}
      onClick={onClick}>
      <div
        className="w-full h-full bg-contain bg-no-repeat bg-center"
        style={{
          backgroundImage: thumbnail == null ? 'none' : `url(${thumbnail})`,
        }}></div>
    </div>
  );
}
