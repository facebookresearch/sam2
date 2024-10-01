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
import PrimaryCTAButton from '@/common/components/button/PrimaryCTAButton';
import useVideo from '@/common/components/video/editor/useVideo';
import {ChevronRight} from '@carbon/icons-react';

type Props = {
  onSessionClose: () => void;
};

export default function CloseSessionButton({onSessionClose}: Props) {
  const video = useVideo();

  function handleCloseSession() {
    video?.closeSession();
    video?.logAnnotations();
    onSessionClose();
  }

  return (
    <PrimaryCTAButton onClick={handleCloseSession} endIcon={<ChevronRight />}>
      Good to go
    </PrimaryCTAButton>
  );
}
