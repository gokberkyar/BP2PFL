const Libp2p = require('libp2p')
const TCP = require('libp2p-tcp')
const Mplex = require('libp2p-mplex')
const { NOISE } = require('@chainsafe/libp2p-noise')
const Gossipsub = require('libp2p-gossipsub')
const Bootstrap = require('libp2p-bootstrap')
const PubsubPeerDiscovery = require('libp2p-pubsub-peer-discovery')
const { fromString } = require('uint8arrays/from-string')
const { toString } = require('uint8arrays/to-string')
const PeerId = require('peer-id')
// const { Resolver } = require('dns').promises;
const spawn = require("child_process").spawn;
const PYFILEPATH = '/app/worker.py'
let dataJson = require('/app/data/peers.json');
const fs  = require('fs')
const MESSAGE_COUNT =dataJson.peers.length -1;
PORT = dataJson.port
const IS_ADVERSARY = dataJson.adversary
const MODEL_POISONING = dataJson.model_poisoning
const ADVERSARY_EPOCHS = dataJson.adversary_epochs
const K_TOLERANT = dataJson.k_tolerant
const CLIPPING_NORM = dataJson.clipping_norm
const LOCAL_NORM = dataJson.local_norm
const DEFENSE = dataJson.defense



console.log(`peer count : ${MESSAGE_COUNT}`)
console.log(`port : ${PORT}`)
var start = new Date().getTime();



const BASE_PORT = 9000;
const keyList = dataJson.keyList;
const personalKey = keyList[dataJson.id];
const TOTAL_NODES =dataJson.nodeCount 
console.log('node count')
console.log(TOTAL_NODES)
// console.log(personalKey)
const NUM_ROUNDS = dataJson.numRounds ;
console.log('total rounds')
console.log(NUM_ROUNDS)

let current_round = 0;
let discoveredPeer = 0;


const createNode = async (bootstrapers) => {
  const peerKeyId = await PeerId.createFromJSON(personalKey)

  const node = await Libp2p.create({

    addresses: {
      listen: ['/ip4/0.0.0.0/tcp/'+PORT]
    },
    peerId:peerKeyId,
    modules: {
      transport: [TCP],
      streamMuxer: [Mplex],
      connEncryption: [NOISE],
      pubsub: Gossipsub,
      // peerDiscovery: [ Bootstrap,PubsubPeerDiscovery]
      peerDiscovery: [ Bootstrap]
    },
    dialer: {
      maxParallelDials: 1000,
      maxAddrsToDial: 100,
      maxDialsPerPeer: 10,
      dialTimeout: 30e3,
    },
    config: {
      connectionManager: {
        maxConnections: Infinity,
        minConnections: 0,
        pollInterval: 2000,
        defaultPeerValue: 1,
        // The below values will only be taken into account when Metrics are enabled
        maxData: Infinity,
        maxSentData: Infinity,
        maxReceivedData: Infinity,
        maxEventLoopDelay: Infinity,
        movingAverageInterval: 60000
      },

      pubsub: {
        enabled: true,

      },
      relay: {
        enabled: false, // Allows you to dial and accept relayed connections. Does not make you a relay.
        hop: {
          enabled: false // Allows you to be a relay for other peers
        }
      },
      peerDiscovery: {
        // [PubsubPeerDiscovery.tag]: {
        //   interval: 100,
        //   enabled: true
        // },
        [Bootstrap.tag]: {
          enabled: true,
          interval: 10e3,
          list: bootstrapers
        }
      }
    }
  })


  console.log(peerKeyId)
  console.log(node.peerId)

  return node
}


const doStuff = (node,round) =>{

  if (handledRounds[round] == 1 ){
    console.log(`round already handled round: ${round}`)
    return
  }
  handledRounds[round] = 1 
  console.log(`handling round: ${round}`)
  round = parseInt(round);
  
  const pythonProcess = spawn('python3',[PYFILEPATH, round, MESSAGE_COUNT, IS_ADVERSARY, MODEL_POISONING, ADVERSARY_EPOCHS, K_TOLERANT,CLIPPING_NORM, DEFENSE,LOCAL_NORM]);

  pythonProcess.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });
  

  pythonProcess.on('exit', (data) => {
      // Do something with the data returned from python script
      console.log(`child process exited with code ${data}`);

      if (data == 0){
        if (round == current_round){
          current_round +=1;
          if (current_round == NUM_ROUNDS){
            var end = new Date().getTime();
            var time = end - start;
            console.log(`Time:  ${time}`)
            process.exit(1)
          }
        }

        round = round +1;
        
        console.log(`sending round ${round}`);

        dataJson.peers.forEach(async element => {
          let minimumPart = Math.min(parseInt(element.substring(4)), parseInt(nodeId)).toString();
          let maximumPart = Math.max(parseInt(element.substring(4)), parseInt(nodeId)).toString();
          let uniqueTuple = minimumPart+'-'+maximumPart;
      
          setTimeout(() => {
              const fileData =  fs.readFileSync('data/mymodel-' + round + '.h5');
              node.pubsub.publish(uniqueTuple+'-'+ round   ,fileData)
      
          }, 10000)
        })

      }

      if (data == 1){
        console.log(`rerunning round ${round}`);
        handledRounds[round] = 0
        setTimeout(() => {
          doStuff(node,round)
        }, 10000)
      }

      




  });

}

const subscribePeers = async (dataJson,node) =>{
  //Subscribe to your neighbours
  dataJson.peers.forEach(async element => {
    let minimumPart = Math.min(parseInt(element.substring(4)), parseInt(nodeId)).toString();
    let maximumPart = Math.max(parseInt(element.substring(4)), parseInt(nodeId)).toString();
    let uniqueTuple = minimumPart+'-'+maximumPart;

    
    // console.log(data.topicIDs)
     console.log(element);

    console.log(uniqueTuple);
    for (let i = 0; i < NUM_ROUNDS; i++) {
      const uniqueTriplet = uniqueTuple+'-'+i;
      node.pubsub.on(uniqueTriplet,  (data) => {
        // console.log(data.topicIDs)
        // console.log(data)
        // console.log('start')
        console.log('receiving from' + uniqueTriplet)
        var path = 'data/'+uniqueTriplet+'.h5'
        if (fs.existsSync(path) == false) {
          //file exists
        

          fs.writeFile(path,data.data, { flag: 'w+' },(err) => {
                  if (err) console.log(err);
                  else {
                          console.log(`The file has been saved! from round ${i} peer: ${element.substring(4)} totalFileRound: ${recevivedMessages[i]} `);
                          incrementReceivedMessages(doStuff,i,node);
                        }
          });
        }
        else{
          console.log('file exists')
        }
  
  
      })
      await node.pubsub.subscribe(uniqueTriplet);
      
    }


  });


}

const incrementReceivedMessages = (callback,index,node) => {
  recevivedMessages[index]  +=1;
  if (recevivedMessages[index] == MESSAGE_COUNT){
    console.log(`callback called for round ${index}`)
    callback(node,index);
  } 
} 




const recevivedMessages =  new Array(NUM_ROUNDS);
for (let index = 0; index < recevivedMessages.length; index++) {
  recevivedMessages[index]=0;
}
const handledRounds =  new Array(NUM_ROUNDS);
for (let index = 0; index < handledRounds.length; index++) {
  handledRounds[index]=0;
}






const main = async () => {
  // const resolver = new Resolver();
  // const url = "relay";
  // const ip = await resolver.resolve4(url);
  // console.log('Relay address')
  // console.log(ip);


  // const relayMultiaddrs = [

  //   '/dns4/172.23.0.2/tcp/8000/p2p/QmUQgf34UYLfPeo65toApiC6mFVA637Kp44fyjhByUXj4V',
  //   // '/ipv4/'+ip[0]+'/tcp/8000/p2p/QmaFKzfQ9Zm7TjfWagCbsNrEwL3XNMZ6Vy1gSqZrugkFeR',
  //   // '/ipv4/'+ip[0]+'/tcp/8000/p2p/QmUQgf34UYLfPeo65toApiC6mFVA637Kp44fyjhByUXj4V',
  // ];


  const bootstrapMultiaddrs =  new Array(TOTAL_NODES);
  for (let index = 0; index < bootstrapMultiaddrs.length; index++) {
    //   // 
    const p2pPart = await PeerId.createFromJSON(keyList[index])
    

    bootstrapMultiaddrs[index]= `/dns4/124.42.0.${1+index}/tcp/${BASE_PORT+index}/p2p/${p2pPart.toB58String()}`
  }
  


  console.log(bootstrapMultiaddrs);
  nodeId = dataJson.id;


  console.log(dataJson.id);


  


  const node= await createNode(bootstrapMultiaddrs)


  // node.on('peer:discovery', (peerId) => {

  //   // console.log(`peer:discovery Peer ${node.peerId.toB58String()} discovered: ${peerId.toB58String()}`)
  //   discoveredPeer +=1;
  //   console.log(`peer:discovery Peer Count ${discoveredPeer}`)



  // });
  // node.connectionManager.on('peer:connect', (connection) => {
  //   console.log('peer:connect')
  //   console.log(connection)
  // })
  node.connectionManager.on('peer:disconnect', (connection) => {
    console.log('peer:disconnect')

    console.log(connection)
  })
  // // node.peerStore.on('peer', (peerId) => {
  //   console.log('peer')

  //   console.log(peerId)

  // })
  // node.peerStore.on('change:multiaddrs', ({ peerId, multiaddrs}) => {
  //   console.log('change:multiaddrs')

  //   console.log(peerId)
  //   console.log(multiaddrs)

  // })





  console.log(`Node starting with id: ${node.peerId.toB58String()}`)
  await node.start()
  await node.pubsub.start();

  setTimeout(async () => {
    await subscribePeers(dataJson,node);
    console.log('Subscription Finished')

  }, 100000)



  dataJson.peers.forEach(async element => {
    let minimumPart = Math.min(parseInt(element.substring(4)), parseInt(nodeId)).toString();
    let maximumPart = Math.max(parseInt(element.substring(4)), parseInt(nodeId)).toString();
    let uniqueTuple = minimumPart+'-'+maximumPart;

    setTimeout(async () => {
        console.log('Sending file ' + uniqueTuple)
        // for (const [peerId, connections] of node.connections) {
        //   for (const connection of connections) {
        //     console.log(peerId, connection.remoteAddr.toString())
        //     // Logs the PeerId string and the observed remote multiaddr of each Connection
        //   }
        // }

        const fileData =  fs.readFileSync('data/mymodel-0.h5');
        // console.log(fileData)
        result = await node.pubsub.publish(uniqueTuple+'-0',fileData)
        // console.log(result)

    }, 300000)

  })




  

   

  
};

main()


