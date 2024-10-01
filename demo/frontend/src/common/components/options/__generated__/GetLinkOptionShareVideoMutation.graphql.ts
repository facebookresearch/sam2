/**
 * @generated SignedSource<<39d7e92a6c15de1583c90ae21a7825e5>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest, Mutation } from 'relay-runtime';
export type GetLinkOptionShareVideoMutation$variables = {
  file: any;
};
export type GetLinkOptionShareVideoMutation$data = {
  readonly uploadSharedVideo: {
    readonly path: string;
  };
};
export type GetLinkOptionShareVideoMutation = {
  response: GetLinkOptionShareVideoMutation$data;
  variables: GetLinkOptionShareVideoMutation$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "defaultValue": null,
    "kind": "LocalArgument",
    "name": "file"
  }
],
v1 = [
  {
    "alias": null,
    "args": [
      {
        "kind": "Variable",
        "name": "file",
        "variableName": "file"
      }
    ],
    "concreteType": "SharedVideo",
    "kind": "LinkedField",
    "name": "uploadSharedVideo",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "kind": "ScalarField",
        "name": "path",
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
    "name": "GetLinkOptionShareVideoMutation",
    "selections": (v1/*: any*/),
    "type": "Mutation",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": (v0/*: any*/),
    "kind": "Operation",
    "name": "GetLinkOptionShareVideoMutation",
    "selections": (v1/*: any*/)
  },
  "params": {
    "cacheID": "f02ec81a41c8d75c3733853e1fb04f58",
    "id": null,
    "metadata": {},
    "name": "GetLinkOptionShareVideoMutation",
    "operationKind": "mutation",
    "text": "mutation GetLinkOptionShareVideoMutation(\n  $file: Upload!\n) {\n  uploadSharedVideo(file: $file) {\n    path\n  }\n}\n"
  }
};
})();

(node as any).hash = "c1b085da9afaac5f19eeb99ff561ed55";

export default node;
