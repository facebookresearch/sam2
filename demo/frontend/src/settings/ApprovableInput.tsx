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
import Tooltip from '@/common/components/Tooltip';
import {ArrowPathIcon, CheckIcon, XMarkIcon} from '@heroicons/react/24/solid';
import {ChangeEvent, KeyboardEvent, useEffect, useMemo, useState} from 'react';
import {Button, Form, Input, Join} from 'react-daisyui';

type Props<T extends string | number> = Omit<
  React.InputHTMLAttributes<HTMLInputElement>,
  'size' | 'color' | 'onChange'
> & {
  label: string;
  defaultValue: T;
  initialValue: T;
  onChange: (value: string) => void;
};

function getStep(value: number) {
  const stringValue = String(value);
  const decimals = stringValue.split('.')[1];
  if (decimals != null) {
    // Not using 0.1 ** decimals.length because this will result in rounding
    // errors, e.g., 0.1 ** 2 => 0.010000000000000002.
    return 1 / 10 ** decimals.length;
  }
  return 1;
}

export default function ApprovableInput<T extends string | number>({
  label,
  defaultValue,
  initialValue,
  onChange,
  ...otherProps
}: Props<T>) {
  const [value, setValue] = useState<string>(`${initialValue}`);

  useEffect(() => {
    setValue(`${initialValue}`);
  }, [initialValue]);

  const step = useMemo(() => {
    return typeof defaultValue === 'number' && isFinite(defaultValue)
      ? getStep(defaultValue)
      : undefined;
  }, [defaultValue]);

  return (
    <div>
      <Form.Label className="flex-col items-start gap-2" title={label}>
        <Join className="w-full">
          <Input
            {...otherProps}
            className="w-full join-item"
            value={value}
            step={step}
            placeholder={`${defaultValue}`}
            onChange={(event: ChangeEvent<HTMLInputElement>) => {
              setValue(event.target.value);
            }}
            onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                onChange(value);
              }
            }}
          />
          <Tooltip message="Reset to default">
            <Button
              className="join-item"
              onClick={event => {
                event.preventDefault();
                setValue(`${defaultValue}`);
              }}>
              <ArrowPathIcon className="h-4 w-4" />
            </Button>
          </Tooltip>
          <Tooltip message="Revert change">
            <Button
              className="join-item"
              color="neutral"
              disabled={initialValue == value}
              onClick={event => {
                event.preventDefault();
                setValue(`${initialValue}`);
              }}>
              <XMarkIcon className="h-4 w-4" />
            </Button>
          </Tooltip>
          <Tooltip message="Apply change">
            <Button
              className="join-item"
              color="primary"
              disabled={initialValue == value}
              onClick={event => {
                event.preventDefault();
                onChange(value);
              }}>
              <CheckIcon className="h-4 w-4" />
            </Button>
          </Tooltip>
        </Join>
      </Form.Label>
    </div>
  );
}
