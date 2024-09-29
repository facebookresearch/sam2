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
import emptyFunction from '@/common/utils/emptyFunction';
import {INFERENCE_API_ENDPOINT, VIDEO_API_ENDPOINT} from '@/demo/DemoConfig';
import SettingsModal from '@/settings/SettingsModal';
import {
  Action,
  DEFAULT_SETTINGS,
  Settings,
  settingsReducer,
} from '@/settings/SettingsReducer';
import {
  PropsWithChildren,
  createContext,
  useCallback,
  useMemo,
  useRef,
} from 'react';
import {useImmerReducer} from 'use-immer';

type ContextProps = {
  settings: Settings;
  dispatch: React.Dispatch<Action>;
  openModal: () => void;
  closeModal: () => void;
  hasChanged: boolean;
};

export const SettingsContext = createContext<ContextProps>({
  settings: DEFAULT_SETTINGS,
  dispatch: emptyFunction,
  openModal: emptyFunction,
  closeModal: emptyFunction,
  hasChanged: false,
});

type Props = PropsWithChildren;

export default function SettingsContextProvider({children}: Props) {
  const [state, dispatch] = useImmerReducer(
    settingsReducer,
    DEFAULT_SETTINGS,
    settings => {
      // Load the settings from local storage. Eventually use the reducer init
      // to handle initial loading.
      return settingsReducer(settings, {type: 'load-state'});
    },
  );

  const modalRef = useRef<HTMLDialogElement>(null);

  const openModal = useCallback(() => {
    modalRef.current?.showModal();
  }, [modalRef]);

  const handleCloseModal = useCallback(() => {
    modalRef.current?.close();
  }, [modalRef]);

  const hasChanged = useMemo(() => {
    return (
      VIDEO_API_ENDPOINT !== state.videoAPIEndpoint ||
      INFERENCE_API_ENDPOINT !== state.inferenceAPIEndpoint
    );
  }, [state.videoAPIEndpoint, state.inferenceAPIEndpoint]);

  const value = useMemo(
    () => ({
      settings: state,
      dispatch,
      openModal,
      closeModal: handleCloseModal,
      hasChanged,
    }),
    [state, dispatch, openModal, handleCloseModal, hasChanged],
  );

  return (
    <SettingsContext.Provider value={value}>
      {children}
      <SettingsModal ref={modalRef} />
    </SettingsContext.Provider>
  );
}
