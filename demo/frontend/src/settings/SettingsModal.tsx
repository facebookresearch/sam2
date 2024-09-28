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
import {DEMO_FRIENDLY_NAME} from '@/demo/DemoConfig';
import SAM2Settings from '@/settings/SAM2Settings';
import {XMarkIcon} from '@heroicons/react/24/solid';
import {forwardRef, useState} from 'react';
import {Button, Modal} from 'react-daisyui';
import useSettingsContext from './useSettingsContext';

type Props = unknown;

type Config = {
  key: 'sam2';
  title: string;
  component: React.ElementType;
};

const SettingsConfig: Config[] = [
  {
    key: 'sam2',
    title: DEMO_FRIENDLY_NAME,
    component: SAM2Settings,
  },
];

export default forwardRef<HTMLDialogElement, Props>(
  function SettingsModal(_props, ref) {
    const {closeModal} = useSettingsContext();
    const [activeConfig, setActiveConfig] = useState<Config>(SettingsConfig[0]);

    const SettingsComponent = activeConfig.component;

    return (
      <Modal
        data-testid="settings-modal"
        ref={ref}
        className="lg:absolute lg:top-10 lg:w-11/12 lg:max-w-4xl flex flex-col"
        responsive={true}>
        <Button
          size="sm"
          color="ghost"
          shape="circle"
          className="absolute right-2 top-2"
          startIcon={<XMarkIcon className="w-6 h-6" />}
          onClick={closeModal}
        />
        <Modal.Header className="font-bold">Settings</Modal.Header>
        <Modal.Body className="flex flex-col grow overflow-hidden">
          <div className="flex flex-col md:lg:flex-row gap-4 md:lg:gap-12 overflow-hidden">
            <div className="flex flex-row shrink-0 md:lg:flex-col gap-4 md:lg:py-2 overflow-x-auto">
              {SettingsConfig.map(config => (
                <div
                  key={config.key}
                  data-testid={`show-settings-${config.key}`}
                  className={`cursor-pointer whitespace-nowrap ${
                    activeConfig.key === config.key && 'text-primary'
                  } ${
                    activeConfig.key === config.key &&
                    'sm:underline md:lg:no-underline sm:underline-offset-4'
                  }`}
                  onClick={() => setActiveConfig(config)}>
                  {config.title}
                </div>
              ))}
            </div>
            <div
              data-testid={`settings-${activeConfig.key}`}
              className="overflow-hidden overflow-y-auto grow md:lg:pt-2">
              <div className="flex flex-col grow-0 flex-1">
                <h1 className="hidden md:lg:block">{activeConfig.title}</h1>
                <SettingsComponent />
              </div>
            </div>
          </div>
        </Modal.Body>
        <Modal.Actions className="shrink-0">
          <Button onClick={closeModal}>Close</Button>
        </Modal.Actions>
      </Modal>
    );
  },
);
