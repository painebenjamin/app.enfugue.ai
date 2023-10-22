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
    /**
     * @var bool Always ask for inclusions
     */
    static alwaysInclude = true;

    /**
     * @var array Related models to include
     */
    static apiInclude = ["refiner", "inpainter", "lora", "lycoris", "inversion", "scheduler", "vae", "config"];

    /**
     * @var string Model root
     */
    static apiRoot = "models";

    /**
     * @var array Scope for an individual item (i.e. primary keys)
     */
    static apiScope = ["name"];

    /**
     * Gets the status for a configured model
     */
    getStatus() {
        return this.queryModel("get", `${this.url}/status`);
    }

    /**
     * Turns the included config into an object.
     */
    get defaultConfiguration() {
        let config = {};
        if (Array.isArray(this.config)) {
            for (let configurationPart of this.config) {
                config[configurationPart.configuration_key] = configurationPart.configuration_value;
            }
        }
        return config;
    }
};

class DiffusionModelRefiner extends ModelObject {};
class DiffusionModelScheduler extends ModelObject {};
class DiffusionModelVAE extends ModelObject {};
class DiffusionModelInpainter extends ModelObject {};
class DiffusionModelInversion extends ModelObject {};
class DiffusionModelLora extends ModelObject {};
class DiffusionModelLycoris extends ModelObject {};
class DiffusionModelDefaultConfiguration extends ModelObject {};

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
        DiffusionModelScheduler,
        DiffusionModelDefaultConfiguration,
        DiffusionModelVAE,
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
