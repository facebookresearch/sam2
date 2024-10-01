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
import ChangeVideoModal from '@/common/components/gallery/ChangeVideoModal';
import type {VideoGalleryTriggerProps} from '@/common/components/gallery/DemoVideoGalleryModal';
import useScreenSize from '@/common/screen/useScreenSize';
import {ImageCopy} from '@carbon/icons-react';
import OptionButton from './OptionButton';

type Props = {
  onChangeVideo: () => void;
};
export default function GalleryOption({onChangeVideo}: Props) {
  return (
    <ChangeVideoModal
      videoGalleryModalTrigger={GalleryTrigger}
      showUploadInGallery={false}
      onChangeVideo={onChangeVideo}
    />
  );
}

function GalleryTrigger({onClick}: VideoGalleryTriggerProps) {
  const {isMobile} = useScreenSize();

  return (
    <OptionButton
      variant="flat"
      title={isMobile ? 'Gallery' : 'Browse gallery'}
      Icon={ImageCopy}
      onClick={onClick}
    />
  );
}
