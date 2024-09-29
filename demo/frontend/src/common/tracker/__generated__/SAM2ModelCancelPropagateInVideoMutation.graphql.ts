/**
 * @generated SignedSource<<87827cb79ef9276cd5a66026151e937c>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type CancelPropagateInVideoInput = {
  sessionId: string;
};
export type SAM2ModelCancelPropagateInVideoMutation$variables = {
  input: CancelPropagateInVideoInput;
};
export type SAM2ModelCancelPropagateInVideoMutation$data = {
  readonly cancelPropagateInVideo: {
    readonly success: boolean;
  };
};
export type SAM2ModelCancelPropagateInVideoMutation = {
  response: SAM2ModelCancelPropagateInVideoMutation$data;
  variables: SAM2ModelCancelPropagateInVideoMutation$variables;
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
    "concreteType": "CancelPropagateInVideo",
    "kind": "LinkedField",
    "name": "cancelPropagateInVideo",
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
    "name": "SAM2ModelCancelPropagateInVideoMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "SAM2ModelCancelPropagateInVideoMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "f00f78f24741d27828f0bd95b0f373c2",
    "id": null,
    "metadata": {},
    "name": "SAM2ModelCancelPropagateInVideoMutation",
    "operationKind": "mutation",
    "text": "mutation SAM2ModelCancelPropagateInVideoMutation(\n  $input: CancelPropagateInVideoInput!\n) {\n  cancelPropagateInVideo(input: $input) {\n    success\n  }\n}\n"
  }
};
})();

(node as any).hash = "1abafecade479ab3c45f9cecf0360285";

export default node;
