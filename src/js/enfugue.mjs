import { Application } from './application/index.mjs';
import { Loader } from './base/loader.mjs';
import { getQueryParameters, isEmpty } from './base/helpers.mjs';

(async () => {
    let parameters = getQueryParameters(),
        globalParameters = window.enfugue || {},
        defaultConfiguration = 'index',
        configurationName,
        configurationModule;

    if (!isEmpty(parameters.config)) {
        try {
            configurationModule = await import(
                `./config/${parameters.config}.mjs`
            );
            configurationName = parameters.config;
        } catch (e) {
            console.error(
                "Couldn't get configuration",
                parameters.config,
                ', using default.'
            );
        }
    }

    if (configurationModule === undefined) {
        configurationName = defaultConfiguration;
        configurationModule = await import(
            `./config/${defaultConfiguration}.mjs`
        );
    }

    let configuration = configurationModule.Configuration;
    configuration.name = configurationName;

    if (!isEmpty(globalParameters.url)) {
        configuration.url = {
            ...(configuration.url || {}), 
            ...globalParameters.url
        };
    }

    if (!isEmpty(globalParameters.keys)) {
        configuration.keys = globalParameters.keys;
    }

    if (parameters.debug) {
        configuration.debug = true;
    }

    const app = new Application(configuration);

    Loader.done(function(){
        app.initialize();
        if (parameters.debug) {
            console.log("Application initialized", app);
        }
    });
})();
