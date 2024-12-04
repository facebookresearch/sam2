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
import Logger from '@/common/logger/Logger';
import {
  CacheConfig,
  GraphQLResponse,
  RequestParameters,
  UploadableMap,
  Variables,
} from 'relay-runtime';

/**
 * Inspired by https://github.com/facebook/relay/issues/1844
 */
export default async function fetchGraphQL(
  endpoint: string,
  request: RequestParameters,
  variables: Variables,
  cacheConfig: CacheConfig,
  uploadables?: UploadableMap | null,
): Promise<GraphQLResponse> {
  const url = `${endpoint}/graphql`;

  const headers: {[name: string]: string} = {};
  const requestInit: RequestInit = {
    method: 'POST',
    headers,
    credentials: 'include',
  };

  const customHeaders = (cacheConfig?.metadata?.headers ?? {}) as {
    [key: string]: string;
  };

  requestInit.headers = Object.assign(customHeaders, requestInit.headers);

  if (uploadables != null) {
    const formData = new FormData();
    formData.append(
      'operations',
      JSON.stringify({
        query: request.text,
        variables,
      }),
    );

    const uploadableMap: {
      [key: string]: string[];
    } = {};

    Object.keys(uploadables).forEach(key => {
      uploadableMap[key] = [`variables.${key}`];
    });

    formData.append('map', JSON.stringify(uploadableMap));

    Object.keys(uploadables).forEach(key => {
      formData.append(key, uploadables[key]);
    });

    requestInit.body = formData;
  } else {
    requestInit.headers = Object.assign(
      {'Content-Type': 'application/json'},
      requestInit.headers,
    );

    requestInit.body = JSON.stringify({
      query: request.text,
      variables,
    });
  }

  try {
    const response = await fetch(url, requestInit);
    const result = await response.json();

    // Handle any intentional GraphQL errors, which are passed through the
    // errors property in the JSON payload.
    if ('errors' in result) {
      for (const error of result.errors) {
        Logger.error(error);
      }
    }

    return result;
  } catch (error) {
    Logger.error(`Could not connect to GraphQL endpoint ${url}`, error);
    throw error;
  }
}
