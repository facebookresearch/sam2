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
import useUploadVideo from '@/common/components/gallery/useUploadVideo';
import useScreenSize from '@/common/screen/useScreenSize';
import {VideoData} from '@/demo/atoms';
import {MAX_UPLOAD_FILE_SIZE} from '@/demo/DemoConfig';
import {BLUE_PINK_FILL_BR} from '@/theme/gradientStyle';
import {RetryFailed, Upload} from '@carbon/icons-react';
import {CSSProperties, ReactNode} from 'react';
import {Loading} from 'react-daisyui';

type Props = {
  style: CSSProperties;
  onUpload: (video: VideoData) => void;
  onUploadStart?: () => void;
  onUploadError?: (error: Error) => void;
};

export default function VideoGalleryUploadVideo({
  style,
  onUpload,
  onUploadStart,
  onUploadError,
}: Props) {
  const {getRootProps, getInputProps, isUploading, error} = useUploadVideo({
    onUpload,
    onUploadStart,
    onUploadError,
  });
  const {isMobile} = useScreenSize();

  return (
    <div className={`cursor-pointer ${BLUE_PINK_FILL_BR}`} style={style}>
      <span {...getRootProps()}>
        <input {...getInputProps()} />
        <div className="relative w-full h-full">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
            {isUploading && (
              <IconWrapper
                icon={
                  <Loading
                    size={isMobile ? 'md' : 'lg'}
                    className="text-white"
                  />
                }
                title="Uploading ..."
              />
            )}
            {error !== null && (
              <IconWrapper
                icon={<RetryFailed color="white" size={isMobile ? 24 : 32} />}
                title={error}
              />
            )}
            {!isUploading && error === null && (
              <IconWrapper
                icon={<Upload color="white" size={isMobile ? 24 : 32} />}
                title={
                  <>
                    Upload{' '}
                    <div className="text-xs opacity-70">
                      Max {MAX_UPLOAD_FILE_SIZE}
                    </div>
                  </>
                }
              />
            )}
          </div>
        </div>
      </span>
    </div>
  );
}

type IconWrapperProps = {
  icon: ReactNode;
  title: ReactNode | string;
};

function IconWrapper({icon, title}: IconWrapperProps) {
  return (
    <>
      <div className="flex justify-center">{icon}</div>
      <div className="mt-1 text-sm md:text-lg text-white font-medium text-center leading-tight">
        {title}
      </div>
    </>
  );
}
