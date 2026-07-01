const d3 = require('d3-force');

let inputData = "";
process.stdin.on('data', chunk => inputData += chunk);
process.stdin.on('end', () => {
    try {
        const graph = JSON.parse(inputData);
        // 【核心修复】：使用对象展开语法浅拷贝，确保 ID 能被正常序列化
        const nodes = graph.nodes.map(d => ({ ...d }));
        const links = graph.edges.map(d => ({ ...d }));

        const simulation = d3.forceSimulation(nodes)
            .force("charge", d3.forceManyBody().strength(-30))
            .force("link", d3.forceLink(links).id(d => d.id).distance(30))
            .force("center", d3.forceCenter(0, 0))
            .stop();

        const n = Math.ceil(Math.log(simulation.alphaMin()) / Math.log(1 - simulation.alphaDecay()));
        for (let i = 0; i < n; ++i) {
            simulation.tick();
        }

        console.log(JSON.stringify({ nodes: nodes }));
    } catch (error) {
        console.error("D3 Calculation Error:", error);
        process.exit(1);
    }
});
