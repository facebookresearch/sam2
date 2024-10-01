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
declare module 'mp4box' {
  export interface MP4MediaTrackEdit {
    media_rate_fraction: number;
    media_rate_integer: number;
    media_time: number;
    segment_duration: number;
  }

  export interface MP4MediaTrack {
    id: number;
    created: Date;
    modified: Date;
    movie_duration: number;
    movie_timescale: number;
    layer: number;
    alternate_group: number;
    volume: number;
    track_width: number;
    track_height: number;
    timescale: number;
    duration: number;
    bitrate: number;
    codec: string;
    language: string;
    nb_samples: number;
    samples_duration: number;
    edits: MP4MediaTrackEdit[];
  }

  export interface MP4VideoData {
    width: number;
    height: number;
  }

  export interface MP4VideoTrack extends MP4MediaTrack {
    video: MP4VideoData;
  }

  export interface MP4AudioData {
    sample_rate: number;
    channel_count: number;
    sample_size: number;
  }

  export interface MP4AudioTrack extends MP4MediaTrack {
    audio: MP4AudioData;
  }

  export type MP4Track = MP4VideoTrack | MP4AudioTrack;

  export interface MP4Info {
    duration: number;
    timescale: number;
    fragment_duration: number;
    isFragmented: boolean;
    isProgressive: boolean;
    hasIOD: boolean;
    brands: string[];
    created: Date;
    modified: Date;
    tracks: MP4Track[];
    audioTracks: MP4AudioTrack[];
    videoTracks: MP4VideoTrack[];
    otherTracks: MP4VideoTrack[];
  }

  export interface MP4Sample {
    alreadyRead: number;
    chunk_index: number;
    chunk_run_index: number;
    cts: number;
    data: Uint8Array;
    degradation_priority: number;
    depends_on: number;
    description: unknown;
    description_index: number;
    dts: number;
    duration: number;
    has_redundancy: number;
    is_depended_on: number;
    is_leading: number;
    is_sync: boolean;
    number: number;
    offset: number;
    size: number;
    timescale: number;
    track_id: number;
  }

  export type MP4ArrayBuffer = ArrayBuffer & {fileStart: number};

  export class DataStream {
    static BIG_ENDIAN: boolean;
    static LITTLE_ENDIAN: boolean;
    buffer: ArrayBuffer;
    constructor(
      arrayBuffer?: ArrayBuffer,
      byteOffset: number,
      endianness: boolean,
    ): void;
  }

  export interface Trak {
    mdia?: {
      minf?: {
        stbl?: {
          stsd?: {
            entries: {
              avcC?: {
                write: (stream: DataStream) => void;
              };
              hvcC?: {
                write: (stream: DataStream) => void;
              };
            }[];
          };
        };
      };
    };
  }

  export namespace BoxParser {
    export class Box {
      size?: number;
      data?: Uint8Array;

      constructor(type?: string, size?: number);

      add(name: string): Box;
      addBox(box: Box): Box;
      addEntry(value: string, prop?: string): void;
      write(stream: DataStream): void;
      writeHeader(stream: DataStream, msg?: string): void;
      computeSize(): void;
    }

    export class ContainerBox extends Box {}

    export class avcCBox extends ContainerBox {}

    export class hvcCBox extends ContainerBox {}

    export class vpcCBox extends ContainerBox {}

    export class av1CBox extends ContainerBox {}
  }

  export interface TrackOptions {
    id?: number;
    type?: string;
    width?: number;
    height?: number;
    duration?: number;
    layer?: number;
    timescale?: number;
    media_duration?: number;
    language?: string;
    hdlr?: string;

    // video
    avcDecoderConfigRecord?: BufferSource;

    // audio
    balance?: number;
    channel_count?: number;
    samplesize?: number;
    samplerate?: number;

    //captions
    namespace?: string;
    schema_location?: string;
    auxiliary_mime_types?: string;

    description?: BoxParser.Box;
    description_boxes?: BoxParser.Box[];

    default_sample_description_index_id?: number;
    default_sample_duration?: number;
    default_sample_size?: number;
    default_sample_flags?: number;
  }

  export interface SampleOptions {
    sample_description_index?: number;
    duration?: number;
    cts?: number;
    dts?: number;
    is_sync?: boolean;
    is_leading?: number;
    depends_on?: number;
    is_depended_on?: number;
    has_redundancy?: number;
    degradation_priority?: number;
  }

  export interface Sample {
    number: number;
    track_id: number;
    timescale: number;
    description_index: number;
    description: {
      avcC?: BoxParser.avcCBox; // h.264
      hvcC?: BoxParser.hvcCBox; // hevc
      vpcC?: BoxParser.vpcCBox; // vp9
      av1C?: BoxParser.av1CBox; // av1
    };
    data: ArrayBuffer;
    size: number;
    alreadyRead?: number;
    duration: number;
    cts: number;
    dts: number;
    is_sync: boolean;
    is_leading?: number;
    depends_on?: number;
    is_depended_on?: number;
    has_redundancy?: number;
    degradation_priority?: number;
    offset?: number;
  }

  export interface MP4File {
    getBuffer(): MP4ArrayBuffer;
    addTrack(options?: TrackOptions): number;
    addSample(
      track: number,
      data: ArrayBuffer,
      options?: SampleOptions,
    ): Sample;
    addSample(
      trackID: number,
      uint8: Uint8Array,
      arg2: {duration: number; is_sync: boolean},
    ): void;
    onMoovStart?: () => void;
    onReady?: (info: MP4Info) => void;
    onError?: (e: string) => void;
    onSamples?: (id: number, user: unknown, samples: MP4Sample[]) => unknown;
    appendBuffer(data: MP4ArrayBuffer): number;
    save(fileName: string): void;
    start(): void;
    stop(): void;
    /**
     * Indicates that the next samples to process (for extraction or
     * segmentation) start at the given time (Number, in seconds) or at the
     * time of the previous Random Access Point (if useRap is true, default
     * is false). Returns the offset in the file of the next bytes to be
     * provided via appendBuffer.
     *
     * @param time - Start at the given time (Number, in seconds)
     * @param useRap - Random Access Point (if useRap is true, default is false)
     * @returns Returns the offset in the file of the next bytes to be provided via appendBuffer.
     */
    seek: (time: number, useRap: boolean) => number;
    flush(): void;
    releaseUsedSamples(trackId: number, sampleNumber: number): void;
    setExtractionOptions(
      trackId: number,
      user?: unknown,
      options?: {nbSamples?: number; rapAlignment?: number},
    ): void;
    getTrackById(trackId: number): Trak;
  }

  export function createFile(): MP4File;

  export {};
}
