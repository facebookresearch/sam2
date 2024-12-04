/**
 * @generated SignedSource<<db1ee50f3027130f61feafb624026897>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type AddPointsInput = {
  clearOldPoints: boolean;
  frameIndex: number;
  labels: ReadonlyArray<number>;
  objectId: number;
  points: ReadonlyArray<ReadonlyArray<number>>;
  sessionId: string;
};
export type SAM2ModelAddNewPointsMutation$variables = {
  input: AddPointsInput;
};
export type SAM2ModelAddNewPointsMutation$data = {
  readonly addPoints: {
    readonly frameIndex: number;
    readonly rleMaskList: ReadonlyArray<{
      readonly objectId: number;
      readonly rleMask: {
        readonly counts: string;
        readonly size: ReadonlyArray<number>;
      };
    }>;
  };
};
export type SAM2ModelAddNewPointsMutation = {
  response: SAM2ModelAddNewPointsMutation$data;
  variables: SAM2ModelAddNewPointsMutation$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "defaultValue": null,
    "kind": "LocalArgument",
    "name": "input"
  }
],
v1 = [
  {
    "alias": null,
    "args": [
      {
        "kind": "Variable",
        "name": "input",
        "variableName": "input"
      }
    ],
    "concreteType": "RLEMaskListOnFrame",
    "kind": "LinkedField",
    "name": "addPoints",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "frameIndex",
        "storageKey": null
      },
      {
        "alias": null,
        "args": null,
        "concreteType": "RLEMaskForObject",
        "kind": "LinkedField",
        "name": "rleMaskList",
        "plural": true,
        "selections": [
          {
            "alias": null,
            "args": null,
            "kind": "ScalarField",
            "name": "objectId",
            "storageKey": null
          },
          {
            "alias": null,
            "args": null,
            "concreteType": "RLEMask",
            "kind": "LinkedField",
            "name": "rleMask",
            "plural": false,
            "selections": [
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "counts",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "size",
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
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Fragment",
    "metadata": null,
    "name": "SAM2ModelAddNewPointsMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "SAM2ModelAddNewPointsMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "dc86527e91907e696683458ed0943d2f",
    "id": null,
    "metadata": {},
    "name": "SAM2ModelAddNewPointsMutation",
    "operationKind": "mutation",
    "text": "mutation SAM2ModelAddNewPointsMutation(\n  $input: AddPointsInput!\n) {\n  addPoints(input: $input) {\n    frameIndex\n    rleMaskList {\n      objectId\n      rleMask {\n        counts\n        size\n      }\n    }\n  }\n}\n"
  }
};
})();

(node as any).hash = "3c96f05877dd91668c1f9e8a3f1203a5";

export default node;
