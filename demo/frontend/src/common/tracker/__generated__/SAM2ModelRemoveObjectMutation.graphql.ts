/**
 * @generated SignedSource<<3d0d7bdc0d4304f08ea91b7df9efeb1f>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type RemoveObjectInput = {
  objectId: number;
  sessionId: string;
};
export type SAM2ModelRemoveObjectMutation$variables = {
  input: RemoveObjectInput;
};
export type SAM2ModelRemoveObjectMutation$data = {
  readonly removeObject: ReadonlyArray<{
    readonly frameIndex: number;
    readonly rleMaskList: ReadonlyArray<{
      readonly objectId: number;
      readonly rleMask: {
        readonly counts: string;
        readonly size: ReadonlyArray<number>;
      };
    }>;
  }>;
};
export type SAM2ModelRemoveObjectMutation = {
  response: SAM2ModelRemoveObjectMutation$data;
  variables: SAM2ModelRemoveObjectMutation$variables;
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
    "name": "removeObject",
    "plural": true,
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
    "name": "SAM2ModelRemoveObjectMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "SAM2ModelRemoveObjectMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "0accbe68b8deea021539365678e58172",
    "id": null,
    "metadata": {},
    "name": "SAM2ModelRemoveObjectMutation",
    "operationKind": "mutation",
    "text": "mutation SAM2ModelRemoveObjectMutation(\n  $input: RemoveObjectInput!\n) {\n  removeObject(input: $input) {\n    frameIndex\n    rleMaskList {\n      objectId\n      rleMask {\n        counts\n        size\n      }\n    }\n  }\n}\n"
  }
};
})();

(node as any).hash = "2dddf010d202332e6e012443cc1d8e55";

export default node;
