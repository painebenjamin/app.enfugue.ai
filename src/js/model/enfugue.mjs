/** @module model/enfugue */
import { Model, ModelObject } from "./index.mjs";

/**
 * Models configure roots/scopes.
 * If a model leaves these unconfigured, it is a known model type, but has no standard REST-style handlers.
 * They can also provide additional methods, if desired. They can have the following vars:
 * @var string $apiRoot The root for this model.
 * @var array<string> $apiScope The scope(s) for this model (the variables that need to be 
 */
class DiffusionModel extends ModelObject {
    static alwaysInclude = true;
    static apiRoot = "models";
    static apiScope = ["name"];
    static apiInclude = ["refiner", "inpainter", "lora", "lycoris", "inversion"];

    getStatus() {
        return this.queryModel("get", `${this.url}/status`);
    }
};

class DiffusionModelRefiner extends ModelObject {};
class DiffusionModelInpainter extends ModelObject {};
class DiffusionModelInversion extends ModelObject {};
class DiffusionModelLora extends ModelObject {};
class DiffusionModelLycoris extends ModelObject {};

class DiffusionInvocation extends ModelObject {
    static apiRoot = "invocation-history";
    static apiScope = ["id"];
};

class User extends ModelObject {
    static alwaysInclude = true;
    static apiRoot = "users";
    static apiScope = ["username"];
    static apiInclude = ["permission_groups", "permission_groups.group"];
};

class UserPermission extends ModelObject {};
class UserPermissionGroup extends ModelObject {};
class PermissionGroup extends ModelObject {};

/**
 * The extended model just register API objects.
 */
class EnfugueModel extends Model {
    static modelObjects = [
        DiffusionModel,
        DiffusionModelRefiner,
        DiffusionModelInpainter,
        DiffusionModelInversion,
        DiffusionModelLora,
        DiffusionModelLycoris,
        DiffusionInvocation,
        User,
        UserPermission,
        UserPermissionGroup,
        PermissionGroup
    ];
}

export { EnfugueModel as Model, ModelObject };
