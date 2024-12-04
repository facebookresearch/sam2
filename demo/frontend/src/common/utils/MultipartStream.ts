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
const decoder = new TextDecoder();
const encoder = new TextEncoder();
const blankLine = encoder.encode('\r\n');

const STATE_BOUNDARY = 0;
const STATE_HEADERS = 1;
const STATE_BODY = 2;

/**
 * Compares two Uint8Array objects for equality.
 * @param {Uint8Array} a
 * @param {Uint8Array} b
 * @return {bool}
 */
function compareArrays(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length != b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

/**
 * Parses a Content-Type into a multipart boundary.
 * @param {string} contentType
 * @return {Uint8Array} boundary line, including preceding -- and trailing \r\n
 */
function getBoundary(contentType: string): Uint8Array | null {
  // Expects the form "multipart/...; boundary=...".
  // This is not a full MIME media type parser but should be good enough.
  const MULTIPART_TYPE = 'multipart/';
  const BOUNDARY_PARAM = '; boundary=';
  if (!contentType.startsWith(MULTIPART_TYPE)) {
    return null;
  }
  const i = contentType.indexOf(BOUNDARY_PARAM, MULTIPART_TYPE.length);
  if (i == -1) {
    return null;
  }
  const suffix = contentType.substring(i + BOUNDARY_PARAM.length);
  return encoder.encode('--' + suffix + '\r\n');
}

/**
 * Creates a multipart stream.
 * @param {string} contentType A Content-Type header.
 * @param {ReadableStream} body The body of a HTTP response.
 * @return {ReadableStream} a stream of {headers: Headers, body: Uint8Array}
 *     objects.
 */
export default function multipartStream(
  contentType: string,
  body: ReadableStream,
): ReadableStream {
  const reader = body.getReader();
  return new ReadableStream({
    async start(controller) {
      // Define the boundary.
      const boundary = getBoundary(contentType);
      if (boundary === null) {
        controller.error(
          new Error(
            'Invalid content type for multipart stream: ' + contentType,
          ),
        );
        return;
      }
      let pos = 0;
      let buf = new Uint8Array(); // buf.slice(pos) has unprocessed data.
      let state = STATE_BOUNDARY;
      let headers: Headers | null = null; // non-null in STATE_HEADERS and STATE_BODY.
      let contentLength: number | null = null; // non-null in STATE_BODY.

      /**
       * Consumes all complete data in buf or raises an Error.
       * May leave incomplete data at buf.slice(pos).
       */
      function processBuf() {
        // The while(true) condition is reqired
        // eslint-disable-next-line no-constant-condition
        while (true) {
          if (boundary === null) {
            controller.error(
              new Error(
                'Invalid content type for multipart stream: ' + contentType,
              ),
            );
            return;
          }
          switch (state) {
            case STATE_BOUNDARY:
              // Read blank lines (if any) then boundary.
              while (
                buf.length >= pos + blankLine.length &&
                compareArrays(buf.slice(pos, pos + blankLine.length), blankLine)
              ) {
                pos += blankLine.length;
              }

              // Check that it starts with a boundary.
              if (buf.length < pos + boundary.length) {
                return;
              }

              if (
                !compareArrays(buf.slice(pos, pos + boundary.length), boundary)
              ) {
                throw new Error('bad part boundary');
              }
              pos += boundary.length;
              state = STATE_HEADERS;
              headers = new Headers();
              break;

            case STATE_HEADERS: {
              const cr = buf.indexOf('\r'.charCodeAt(0), pos);
              if (cr == -1 || buf.length == cr + 1) {
                return;
              }
              if (buf[cr + 1] != '\n'.charCodeAt(0)) {
                throw new Error('bad part header line (CR without NL)');
              }
              const line = decoder.decode(buf.slice(pos, cr));
              pos = cr + 2;
              if (line == '') {
                const rawContentLength = headers?.get('Content-Length');
                if (rawContentLength == null) {
                  throw new Error('missing/invalid part Content-Length');
                }
                contentLength = parseInt(rawContentLength, 10);
                if (isNaN(contentLength)) {
                  throw new Error('missing/invalid part Content-Length');
                }
                state = STATE_BODY;
                break;
              }
              const colon = line.indexOf(':');
              const name = line.substring(0, colon);
              if (colon == line.length || line[colon + 1] != ' ') {
                throw new Error('bad part header line (no ": ")');
              }
              const value = line.substring(colon + 2);
              headers?.append(name, value);
              break;
            }
            case STATE_BODY: {
              if (contentLength === null) {
                throw new Error('content length not set');
              }
              if (buf.length < pos + contentLength) {
                return;
              }
              const body = buf.slice(pos, pos + contentLength);
              pos += contentLength;
              controller.enqueue({
                headers: headers,
                body: body,
              });
              headers = null;
              contentLength = null;
              state = STATE_BOUNDARY;
              break;
            }
          }
        }
      }

      // The while(true) condition is required
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const {done, value} = await reader.read();
        const buffered = buf.length - pos;
        if (done) {
          if (state != STATE_BOUNDARY || buffered > 0) {
            throw Error('multipart stream ended mid-part');
          }
          controller.close();
          return;
        }

        // Update buf.slice(pos) to include the new data from value.
        if (buffered == 0) {
          buf = value;
        } else {
          const newLen = buffered + value.length;
          const newBuf = new Uint8Array(newLen);
          newBuf.set(buf.slice(pos), 0);
          newBuf.set(value, buffered);
          buf = newBuf;
        }
        pos = 0;

        processBuf();
      }
    },
    cancel(reason) {
      return body.cancel(reason);
    },
  });
}
