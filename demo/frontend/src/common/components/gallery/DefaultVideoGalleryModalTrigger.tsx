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
import ResponsiveButton from '@/common/components/button/ResponsiveButton';
import type {VideoGalleryTriggerProps} from '@/common/components/gallery/DemoVideoGalleryModal';
import {ImageCopy} from '@carbon/icons-react';

export default function DefaultVideoGalleryModalTrigger({
  onClick,
}: VideoGalleryTriggerProps) {
  return (
    <ResponsiveButton
      color="ghost"
      className="hover:!bg-black"
      startIcon={<ImageCopy size={20} />}
      onClick={onClick}>
      Change video
    </ResponsiveButton>
  );
}
