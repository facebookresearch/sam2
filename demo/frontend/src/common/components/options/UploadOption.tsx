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
import OptionButton from '@/common/components/options/OptionButton';
import Logger from '@/common/logger/Logger';
import useScreenSize from '@/common/screen/useScreenSize';
import {sessionAtom, uploadingStateAtom} from '@/demo/atoms';
import {MAX_UPLOAD_FILE_SIZE} from '@/demo/DemoConfig';
import {Close, CloudUpload} from '@carbon/icons-react';
import {useSetAtom} from 'jotai';
import {useNavigate} from 'react-router-dom';

type Props = {
  onUpload: () => void;
};

export default function UploadOption({onUpload}: Props) {
  const navigate = useNavigate();
  const {isMobile} = useScreenSize();
  const setUploadingState = useSetAtom(uploadingStateAtom);
  const setSession = useSetAtom(sessionAtom);

  const {getRootProps, getInputProps, isUploading, error} = useUploadVideo({
    onUpload: videoData => {
      navigate(
        {pathname: location.pathname, search: location.search},
        {state: {video: videoData}},
      );
      onUpload();
      setUploadingState('default');
      setSession(null);
    },
    onUploadError: (error: Error) => {
      setUploadingState('error');
      Logger.error(error);
    },
    onUploadStart: () => {
      setUploadingState('uploading');
    },
  });

  return (
    <div className="cursor-pointer" {...getRootProps()}>
      <input {...getInputProps()} />

      <OptionButton
        variant="gradient"
        title={
          error !== null ? (
            'Upload Error'
          ) : isMobile ? (
            <>
              Upload{' '}
              <div className="text-xs opacity-70">
                Max {MAX_UPLOAD_FILE_SIZE}
              </div>
            </>
          ) : (
            <>
              Upload your own{' '}
              <div className="text-xs opacity-70">
                Max {MAX_UPLOAD_FILE_SIZE}
              </div>
            </>
          )
        }
        Icon={error !== null ? Close : CloudUpload}
        loadingProps={{loading: isUploading, label: 'Uploading...'}}
        onClick={() => {}}
      />
    </div>
  );
}
