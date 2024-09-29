/**
 * @generated SignedSource<<db7e183e1996cf656749b4e33c2424e6>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Query } from 'relay-runtime';
import { FragmentRefs } from "relay-runtime";
export type DemoVideoGalleryModalQuery$variables = Record<PropertyKey, never>;
export type DemoVideoGalleryModalQuery$data = {
  readonly " $fragmentSpreads": FragmentRefs<"DatasetsDropdown_datasets" | "VideoGallery_videos">;
};
export type DemoVideoGalleryModalQuery = {
  response: DemoVideoGalleryModalQuery$data;
  variables: DemoVideoGalleryModalQuery$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "alias": null,
    "args": null,
    "kind": "ScalarField",
    "name": "name",
    "storageKey": null
  }
],
v1 = [
  {
    "kind": "Literal",
    "name": "after",
    "value": ""
  },
  {
    "kind": "Literal",
    "name": "first",
    "value": 20
  }
],
v2 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "__typename",
  "storageKey": null
};
return {
  "fragment": {
    "argumentDefinitions": [],
    "kind": "Fragment",
    "metadata": null,
    "name": "DemoVideoGalleryModalQuery",
    "selections": [
      {
        "args": null,
        "kind": "FragmentSpread",
        "name": "DatasetsDropdown_datasets"
      },
      {
        "args": null,
        "kind": "FragmentSpread",
        "name": "VideoGallery_videos"
      }
    ],
    "type": "Query",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": [],
    "kind": "Operation",
    "name": "DemoVideoGalleryModalQuery",
    "selections": [
      {
        "alias": null,
        "args": null,
        "concreteType": "DatasetConnection",
        "kind": "LinkedField",
        "name": "datasets",
        "plural": false,
        "selections": [
          {
            "alias": null,
            "args": null,
            "concreteType": "DatasetEdge",
            "kind": "LinkedField",
            "name": "edges",
            "plural": true,
            "selections": [
              {
                "alias": null,
                "args": null,
                "concreteType": "Dataset",
                "kind": "LinkedField",
                "name": "node",
                "plural": false,
                "selections": (v0/*: any*/),
                "storageKey": null
              }
            ],
            "storageKey": null
          }
        ],
        "storageKey": null
      },
      {
        "alias": null,
        "args": (v1/*: any*/),
        "concreteType": "VideoConnection",
        "kind": "LinkedField",
        "name": "videos",
        "plural": false,
        "selections": [
          (v2/*: any*/),
          {
            "alias": null,
            "args": null,
            "concreteType": "PageInfo",
            "kind": "LinkedField",
            "name": "pageInfo",
            "plural": false,
            "selections": [
              (v2/*: any*/),
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "hasPreviousPage",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "hasNextPage",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "startCursor",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "endCursor",
                "storageKey": null
              }
            ],
            "storageKey": null
          },
          {
            "alias": null,
            "args": null,
            "concreteType": "VideoEdge",
            "kind": "LinkedField",
            "name": "edges",
            "plural": true,
            "selections": [
              (v2/*: any*/),
              {
                "alias": null,
                "args": null,
                "concreteType": "Video",
                "kind": "LinkedField",
                "name": "node",
                "plural": false,
                "selections": [
                  (v2/*: any*/),
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
                    "name": "width",
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
                    "concreteType": "Dataset",
                    "kind": "LinkedField",
                    "name": "dataset",
                    "plural": false,
                    "selections": (v0/*: any*/),
                    "storageKey": null
                  },
                  {
                    "alias": null,
                    "args": null,
                    "concreteType": "VideoPermissions",
                    "kind": "LinkedField",
                    "name": "permissions",
                    "plural": false,
                    "selections": [
                      {
                        "alias": null,
                        "args": null,
                        "kind": "ScalarField",
                        "name": "canShare",
                        "storageKey": null
                      },
                      {
                        "alias": null,
                        "args": null,
                        "kind": "ScalarField",
                        "name": "canDownload",
                        "storageKey": null
                      }
                    ],
                    "storageKey": null
                  }
                ],
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "cursor",
                "storageKey": null
              }
            ],
            "storageKey": null
          }
        ],
        "storageKey": "videos(after:\"\",first:20)"
      },
      {
        "alias": null,
        "args": (v1/*: any*/),
        "filters": [
          "datasetName"
        ],
        "handle": "connection",
        "key": "VideoGallery_videos",
        "kind": "LinkedHandle",
        "name": "videos"
      }
    ]
  },
  "params": {
    "cacheID": "e0bccf553377682e6bc283c2ce53bee5",
    "id": null,
    "metadata": {},
    "name": "DemoVideoGalleryModalQuery",
    "operationKind": "query",
    "text": "query DemoVideoGalleryModalQuery {\n  ...DatasetsDropdown_datasets\n  ...VideoGallery_videos\n}\n\nfragment DatasetsDropdown_datasets on Query {\n  datasets {\n    edges {\n      node {\n        name\n      }\n    }\n  }\n}\n\nfragment VideoGallery_videos on Query {\n  videos(first: 20, after: \"\") {\n    __typename\n    pageInfo {\n      __typename\n      hasPreviousPage\n      hasNextPage\n      startCursor\n      endCursor\n    }\n    edges {\n      __typename\n      node {\n        __typename\n        id\n        path\n        posterPath\n        url\n        posterUrl\n        width\n        height\n        dataset {\n          name\n        }\n        permissions {\n          canShare\n          canDownload\n        }\n      }\n      cursor\n    }\n  }\n}\n"
  }
};
})();

(node as any).hash = "d09e34e2b9f2e25c2d564106de5f9c89";

export default node;
