/**
 * @generated SignedSource<<f56872c0a8b65fa7e9bdaff351930ff0>>
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
export type useCloseSessionBeforeUnloadMutation$variables = {
  input: CloseSessionInput;
};
export type useCloseSessionBeforeUnloadMutation$data = {
  readonly closeSession: {
    readonly success: boolean;
  };
};
export type useCloseSessionBeforeUnloadMutation = {
  response: useCloseSessionBeforeUnloadMutation$data;
  variables: useCloseSessionBeforeUnloadMutation$variables;
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
    "name": "useCloseSessionBeforeUnloadMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "useCloseSessionBeforeUnloadMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "99b73bd43a9f74104d545778cebbd15c",
    "id": null,
    "metadata": {},
    "name": "useCloseSessionBeforeUnloadMutation",
    "operationKind": "mutation",
    "text": "mutation useCloseSessionBeforeUnloadMutation(\n  $input: CloseSessionInput!\n) {\n  closeSession(input: $input) {\n    success\n  }\n}\n"
  }
};
})();

(node as any).hash = "55dd870645c9736b797b90819ddb1b92";

export default node;
