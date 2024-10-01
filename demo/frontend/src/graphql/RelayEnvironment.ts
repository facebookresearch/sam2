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
  Environment,
  FetchFunction,
  GraphQLResponse,
  LogEvent,
  Network,
  ObservableFromValue,
  RecordSource,
  RequestParameters,
  Store,
  UploadableMap,
  Variables,
} from 'relay-runtime';
import fetchGraphQL from './fetchGraphQL';

function createFetchRelay(endpoint: string): FetchFunction {
  return (
    request: RequestParameters,
    variables: Variables,
    cacheConfig: CacheConfig,
    uploadables?: UploadableMap | null,
  ): ObservableFromValue<GraphQLResponse> => {
    Logger.debug(
      `fetching query ${request.name} with ${JSON.stringify(variables)}`,
    );
    return fetchGraphQL(endpoint, request, variables, cacheConfig, uploadables);
  };
}

export function createEnvironment(endpoint: string): Environment {
  return new Environment({
    log: (logEvent: LogEvent) => Logger.debug(logEvent.name, logEvent),
    network: Network.create(createFetchRelay(endpoint)),
    store: new Store(new RecordSource()),
  });
}
