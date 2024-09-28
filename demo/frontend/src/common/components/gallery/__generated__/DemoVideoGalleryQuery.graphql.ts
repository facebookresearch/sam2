/**
 * @generated SignedSource<<20d31a82b5f3b251b0e42b4f0e3522b8>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Query } from 'relay-runtime';
export type DemoVideoGalleryQuery$variables = Record<PropertyKey, never>;
export type DemoVideoGalleryQuery$data = {
  readonly videos: {
    readonly edges: ReadonlyArray<{
      readonly node: {
        readonly height: number;
        readonly id: any;
        readonly path: string;
        readonly posterPath: string | null | undefined;
        readonly posterUrl: string;
        readonly url: string;
        readonly width: number;
      };
    }>;
  };
};
export type DemoVideoGalleryQuery = {
  response: DemoVideoGalleryQuery$data;
  variables: DemoVideoGalleryQuery$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "alias": null,
    "args": null,
    "concreteType": "VideoConnection",
    "kind": "LinkedField",
    "name": "videos",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "concreteType": "VideoEdge",
        "kind": "LinkedField",
        "name": "edges",
        "plural": true,
        "selections": [
          {
            "alias": null,
            "args": null,
            "concreteType": "Video",
            "kind": "LinkedField",
            "name": "node",
            "plural": false,
            "selections": [
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "id",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "path",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "posterPath",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "url",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "posterUrl",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "height",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "width",
                "storageKey": null
              }
            ],
            "storageKey": null
          }
        ],
        "storageKey": null
      }
    ],
    "storageKey": null
  }
];
return {
  "fragment": {
    "argumentDefinitions": [],
    "kind": "Fragment",
    "metadata": null,
    "name": "DemoVideoGalleryQuery",
    "selections": (v0/*: any*/),
    "type": "Query",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": [],
    "kind": "Operation",
    "name": "DemoVideoGalleryQuery",
    "selections": (v0/*: any*/)
  },
  "params": {
    "cacheID": "4dae74153a5528f2631b59dfb0adb021",
    "id": null,
    "metadata": {},
    "name": "DemoVideoGalleryQuery",
    "operationKind": "query",
    "text": "query DemoVideoGalleryQuery {\n  videos {\n    edges {\n      node {\n        id\n        path\n        posterPath\n        url\n        posterUrl\n        height\n        width\n      }\n    }\n  }\n}\n"
  }
};
})();

(node as any).hash = "d22ac5e58f6e4eb696651be49b410e4e";

export default node;
