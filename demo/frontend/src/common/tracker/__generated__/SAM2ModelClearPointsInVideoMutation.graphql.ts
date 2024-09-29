/**
 * @generated SignedSource<<092c43655450b8af706e546837e0a01c>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type ClearPointsInVideoInput = {
  sessionId: string;
};
export type SAM2ModelClearPointsInVideoMutation$variables = {
  input: ClearPointsInVideoInput;
};
export type SAM2ModelClearPointsInVideoMutation$data = {
  readonly clearPointsInVideo: {
    readonly success: boolean;
  };
};
export type SAM2ModelClearPointsInVideoMutation = {
  response: SAM2ModelClearPointsInVideoMutation$data;
  variables: SAM2ModelClearPointsInVideoMutation$variables;
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
    "concreteType": "ClearPointsInVideo",
    "kind": "LinkedField",
    "name": "clearPointsInVideo",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "success",
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
    "name": "SAM2ModelClearPointsInVideoMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "SAM2ModelClearPointsInVideoMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "c23b3d5afca5b235328a562369056527",
    "id": null,
    "metadata": {},
    "name": "SAM2ModelClearPointsInVideoMutation",
    "operationKind": "mutation",
    "text": "mutation SAM2ModelClearPointsInVideoMutation(\n  $input: ClearPointsInVideoInput!\n) {\n  clearPointsInVideo(input: $input) {\n    success\n  }\n}\n"
  }
};
})();

(node as any).hash = "020267989385cb8b8f0e5cdde784d17e";

export default node;
