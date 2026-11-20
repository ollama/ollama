import { DefaultPackageManager, ModelRuntime, ProjectTrustStore, SettingsManager, getAgentDir, parseArgs } from "@earendil-works/pi-coding-agent";
import { readFileSync, realpathSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";

function isPlugin(path) {
    if (!path) return false;
    try {
        const target = realpathSync(path);
        const root = statSync(target).isDirectory() ? target : dirname(target);
        return JSON.parse(readFileSync(join(root, "package.json"), "utf8")).name === "@ollama/pi";
    } catch {
        return false;
    }
}

function checkSettings(settings) {
    for (const {path, error} of settings.drainErrors()) {
        throw new Error(`${path}: ${error.message}`);
    }
}

async function hasPlugin(packages, cwd, agentDir, plugin, explicit) {
    const declared = packages.listConfiguredPackages();
    if (declared.some(pkg => isPlugin(pkg.installedPath))) return true;
    // Ask Pi to match the package identity, including pins, in a disposable
    // copy of the declarations. A missing checkout still counts as configured.
    const settings = SettingsManager.inMemory({packages: [...declared.map(pkg => pkg.source), ...explicit]});
    const copy = new DefaultPackageManager({cwd, agentDir, settingsManager: settings});
    if (copy.removeSourceFromSettings(plugin) || copy.removeSourceFromSettings("npm:@ollama/pi")) return true;
    const {extensions} = await packages.resolve();
    if (extensions.some(entry => isPlugin(entry.path))) return true;
    const temporary = await packages.resolveExtensionSources(explicit, {temporary: true});
    return temporary.extensions.some(entry => isPlugin(entry.path));
}

async function configure({cwd, agentDir: override, model, args, offline, plugin, configureOnly}) {
    process.chdir(cwd);
    if (override) process.env.PI_CODING_AGENT_DIR = override;
    else delete process.env.PI_CODING_AGENT_DIR;
    const agentDir = getAgentDir();
    const options = parseArgs(args ?? []);
    const settings = SettingsManager.create(cwd, agentDir, {projectTrusted: false});
    settings.setProjectTrusted(options.projectTrustOverride ?? new ProjectTrustStore(agentDir).get(cwd)
        ?? (settings.getDefaultProjectTrust() === "always"));
    if (configureOnly) {
        checkSettings(settings);
    } else if (settings.drainErrors().length) {
        return; // The session will report Pi's settings diagnostic.
    }
    const packages = new DefaultPackageManager({cwd, agentDir, settingsManager: settings});
    if (!options.noExtensions && !await hasPlugin(packages, cwd, agentDir, plugin, options.extensions ?? [])) {
        if (offline) {
            packages.addSourceToSettings(plugin);
        } else {
            delete process.env.PI_OFFLINE;
            await packages.installAndPersist(plugin);
        }
    }
    let id = model;
    if (id) {
        // Pi looks up a saved model by its exact ID. Local Ollama models
        // have a tag, but a user-defined Pi model can have an untagged ID.
        if (!id.startsWith("cloud/") && !basename(id).includes(":")) {
            const catalog = await ModelRuntime.create({refreshOnCreate: false});
            if (!catalog.getModel("ollama", id)) id += ":latest";
        }
        settings.setDefaultModelAndProvider("ollama", id);
        const enabled = settings.getGlobalSettings().enabledModels;
        const reference = `ollama/${id}`;
        if (enabled?.length && !enabled.includes(reference)) settings.setEnabledModels([...enabled, reference]);
    }
    await settings.flush();
    checkSettings(settings);
}

export default async function () {
    try {
        await configure(JSON.parse(readFileSync(0, "utf8")));
        writeFileSync(1, "ollama-pi-ready\n");
        process.exit(0);
    } catch (error) {
        writeFileSync(2, `${error.message}\n`);
        process.exit(1);
    }
}
