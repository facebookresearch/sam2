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
import CreateFilmstripError from '@/graphql/errors/CreateFilmstripError';
import DrawFrameError from '@/graphql/errors/DrawFrameError';
import WebGLContextError from '@/graphql/errors/WebGLContextError';
import {deserializeError, type ErrorObject} from 'serialize-error';

export type RenderingErrorType =
  | 'webgl_context'
  | 'draw_frame'
  | 'create_filmstrip'
  | 'error';

export function getRenderErrorType(error?: ErrorObject): RenderingErrorType {
  const deserializedError = deserializeError(error);

  if (deserializedError instanceof WebGLContextError) {
    return 'webgl_context';
  }
  if (deserializedError instanceof DrawFrameError) {
    return 'draw_frame';
  }
  if (deserializedError instanceof CreateFilmstripError) {
    return 'create_filmstrip';
  }
  return 'error';
}

/**
 * This function extracts the title from an error message.
 * The title is defined as the text before the first newline character.
 *
 * @param error The error object from which the title is to be extracted.
 * @returns The title of the error message.
 * @example
 * ```ts
 * const error = new Error('This is the title\nThis is the body');
 * const title = getErrorTitle(error);
 * console.log(title); // 'This is the title'
 * ```
 */
export function getErrorTitle({message}: Error): string {
  const idx = message.indexOf('\n');
  return idx < 0 ? message : message.substring(0, idx);
}
