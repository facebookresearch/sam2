/**
 * @generated SignedSource<<90910bae5bb646118174e736434aac56>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type StartSessionInput = {
  path: string;
};
export type SAM2ModelStartSessionMutation$variables = {
  input: StartSessionInput;
};
export type SAM2ModelStartSessionMutation$data = {
  readonly startSession: {
    readonly sessionId: string;
  };
};
export type SAM2ModelStartSessionMutation = {
  response: SAM2ModelStartSessionMutation$data;
  variables: SAM2ModelStartSessionMutation$variables;
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
    "concreteType": "StartSession",
    "kind": "LinkedField",
    "name": "startSession",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "sessionId",
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
    "name": "SAM2ModelStartSessionMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "SAM2ModelStartSessionMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "2403f5005f5bb3805109874569f2050e",
    "id": null,
    "metadata": {},
    "name": "SAM2ModelStartSessionMutation",
    "operationKind": "mutation",
    "text": "mutation SAM2ModelStartSessionMutation(\n  $input: StartSessionInput!\n) {\n  startSession(input: $input) {\n    sessionId\n  }\n}\n"
  }
};
})();

(node as any).hash = "5cf0005c7a54fc87c539dd4cbd5fef5d";

export default node;
