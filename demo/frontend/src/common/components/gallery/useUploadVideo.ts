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
import {useUploadVideoMutation} from '@/common/components/gallery/__generated__/useUploadVideoMutation.graphql';
import Logger from '@/common/logger/Logger';
import {VideoData} from '@/demo/atoms';
import {useState} from 'react';
import {FileRejection, FileWithPath, useDropzone} from 'react-dropzone';
import {graphql, useMutation} from 'react-relay';

const ACCEPT_VIDEOS = {
  'video/mp4': ['.mp4'],
  'video/quicktime': ['.mov'],
};

// 70 MB default max video upload size
const MAX_FILE_SIZE_IN_MB = 70;
const MAX_VIDEO_UPLOAD_SIZE = MAX_FILE_SIZE_IN_MB * 1024 ** 2;

type Props = {
  onUpload: (video: VideoData) => void;
  onUploadStart?: () => void;
  onUploadError?: (error: Error) => void;
};

export default function useUploadVideo({
  onUpload,
  onUploadStart,
  onUploadError,
}: Props) {
  const [error, setError] = useState<string | null>(null);
  const [commit, isMutationInFlight] = useMutation<useUploadVideoMutation>(
    graphql`
      mutation useUploadVideoMutation($file: Upload!) {
        uploadVideo(file: $file) {
          id
          height
          width
          url
          path
          posterPath
          posterUrl
        }
      }
    `,
  );

  const {getRootProps, getInputProps} = useDropzone({
    accept: ACCEPT_VIDEOS,
    multiple: false,
    maxFiles: 1,
    onDrop: (
      acceptedFiles: FileWithPath[],
      fileRejections: FileRejection[],
    ) => {
      setError(null);

      // Check if any of the files (only 1 file allowed) is rejected. The
      // rejected file has an error (e.g., 'file-too-large'). Rendering an
      // appropriate message.
      if (fileRejections.length > 0 && fileRejections[0].errors.length > 0) {
        const code = fileRejections[0].errors[0].code;
        if (code === 'file-too-large') {
          setError(
            `File too large. Try a video under ${MAX_FILE_SIZE_IN_MB} MB`,
          );
          return;
        }
      }

      if (acceptedFiles.length === 0) {
        setError('File not accepted. Please try again.');
        return;
      }
      if (acceptedFiles.length > 1) {
        setError('Too many files. Please try again with 1 file.');
        return;
      }

      onUploadStart?.();
      const file = acceptedFiles[0];

      commit({
        variables: {
          file,
        },
        uploadables: {
          file,
        },
        onCompleted: response => onUpload(response.uploadVideo),
        onError: error => {
          Logger.error(error);
          onUploadError?.(error);
          setError('Upload failed.');
        },
      });
    },
    onError: error => {
      Logger.error(error);
      setError('File not supported.');
    },
    maxSize: MAX_VIDEO_UPLOAD_SIZE,
  });

  return {
    getRootProps,
    getInputProps,
    isUploading: isMutationInFlight,
    error,
    setError,
  };
}
