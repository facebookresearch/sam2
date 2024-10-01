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
import useScreenSize from '@/common/screen/useScreenSize';
import {color, gradients} from '@/theme/tokens.stylex';
import {Close} from '@carbon/icons-react';
import stylex from '@stylexjs/stylex';
import {useAtomValue} from 'jotai';
import {Loading, RadialProgress} from 'react-daisyui';
import {messageAtom} from './snackbarAtoms';
import useExpireMessage from './useExpireMessage';
import useMessagesSnackbar from './useMessagesSnackbar';

const styles = stylex.create({
  container: {
    position: 'absolute',
    top: '8px',
    right: '8px',
  },
  mobileContainer: {
    position: 'absolute',
    bottom: '8px',
    left: '8px',
    right: '8px',
  },
  messageContainer: {
    padding: '20px 20px',
    color: '#FFF',
    borderRadius: '8px',
    fontSize: '0.9rem',
    maxWidth: 400,
    border: '2px solid transparent',
    background: gradients['yellowTeal'],
  },
  messageWarningContainer: {
    background: '#FFDC32',
    color: color['gray-900'],
  },
  messageContent: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  progress: {
    flexShrink: 0,
    color: 'rgba(255, 255, 255, 0.1)',
  },
  closeColumn: {
    display: 'flex',
    alignSelf: 'stretch',
    alignItems: 'start',
  },
});

export default function MessagesSnackbar() {
  const message = useAtomValue(messageAtom);
  const {clearMessage} = useMessagesSnackbar();
  const {isMobile} = useScreenSize();

  useExpireMessage();

  if (message == null) {
    return null;
  }

  const closeIcon = (
    <Close
      size={24}
      color={message.type === 'warning' ? color['gray-900'] : 'white'}
      opacity={1}
      className="z-20 hover:text-gray-300 color-white cursor-pointer !opacity-100 shrink-0"
      onClick={clearMessage}
    />
  );

  return (
    <div
      {...stylex.props(isMobile ? styles.mobileContainer : styles.container)}>
      <div
        {...stylex.props(
          styles.messageContainer,
          message.type === 'warning' && styles.messageWarningContainer,
        )}>
        <div {...stylex.props(styles.messageContent)}>
          <div>{message.text}</div>
          {message.type === 'loading' && <Loading size="xs" variant="dots" />}
          {message.showClose && (
            <div {...stylex.props(styles.closeColumn)}>
              {message.expire ? (
                <RadialProgress
                  value={message.progress * 100}
                  size="32px"
                  thickness="2px"
                  {...stylex.props(styles.progress)}>
                  {closeIcon}
                </RadialProgress>
              ) : (
                closeIcon
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
