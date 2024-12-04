/**
 * @generated SignedSource<<48ee5db240b8093e9e53bf0329c8bab7>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type CloseSessionInput = {
  sessionId: string;
};
export type SAM2ModelCloseSessionMutation$variables = {
  input: CloseSessionInput;
};
export type SAM2ModelCloseSessionMutation$data = {
  readonly closeSession: {
    readonly success: boolean;
  };
};
export type SAM2ModelCloseSessionMutation = {
  response: SAM2ModelCloseSessionMutation$data;
  variables: SAM2ModelCloseSessionMutation$variables;
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
    "concreteType": "CloseSession",
    "kind": "LinkedField",
    "name": "closeSession",
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
    "name": "SAM2ModelCloseSessionMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "SAM2ModelCloseSessionMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "aa7177838c16536b397bfee2d15a94ee",
    "id": null,
    "metadata": {},
    "name": "SAM2ModelCloseSessionMutation",
    "operationKind": "mutation",
    "text": "mutation SAM2ModelCloseSessionMutation(\n  $input: CloseSessionInput!\n) {\n  closeSession(input: $input) {\n    success\n  }\n}\n"
  }
};
})();

(node as any).hash = "6e1008de944562dc1922cd3f9cc40f10";

export default node;
