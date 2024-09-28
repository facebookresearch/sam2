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
import useRestartSession from '@/common/components/session/useRestartSession';
import {Reset} from '@carbon/icons-react';
import {Button, Loading} from 'react-daisyui';

type Props = {
  onRestartSession: () => void;
};

export default function RestartSessionButton({onRestartSession}: Props) {
  const {restartSession, isLoading} = useRestartSession();

  function handleRestartSession() {
    restartSession(onRestartSession);
  }

  return (
    <Button
      color="ghost"
      onClick={handleRestartSession}
      className="!px-4 !rounded-full font-medium text-white hover:bg-black"
      startIcon={isLoading ? <Loading size="sm" /> : <Reset size={20} />}>
      Start over
    </Button>
  );
}
