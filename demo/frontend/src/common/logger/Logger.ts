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
import {LOG_LEVEL} from './LogEnvironment';

/** Signature of a logging function */
export type LogFn = {
  (message?: unknown, ...optionalParams: unknown[]): void;
};

/** Basic logger interface */
export interface Logger {
  info: LogFn;
  warn: LogFn;
  error: LogFn;
  debug: LogFn;
}

/** Log levels */
export type LogLevel = 'info' | 'warn' | 'error' | 'debug';

const NO_OP: LogFn = (_message?: unknown, ..._optionalParams: unknown[]) => {};

/** Logger which outputs to the browser console */
export class ConsoleLogger implements Logger {
  readonly info: LogFn;
  readonly warn: LogFn;
  readonly error: LogFn;
  readonly debug: LogFn;

  constructor(options?: {level?: LogLevel}) {
    const {level} = options || {};

    // eslint-disable-next-line no-console
    this.error = console.error.bind(console);

    if (level === 'error') {
      this.debug = NO_OP;
      this.warn = NO_OP;
      this.info = NO_OP;

      return;
    }

    // eslint-disable-next-line no-console
    this.warn = console.warn.bind(console);

    if (level === 'warn') {
      this.debug = NO_OP;
      this.info = NO_OP;

      return;
    }

    // eslint-disable-next-line no-console
    this.info = console.log.bind(console);

    if (level === 'info') {
      this.debug = NO_OP;

      return;
    }

    // eslint-disable-next-line no-console
    this.debug = console.debug.bind(console);
  }
}

export default new ConsoleLogger({level: LOG_LEVEL});
