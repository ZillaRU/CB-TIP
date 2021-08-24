from abc import ABCMeta, abstractproperty, abstractmethod
from collections import Mapping

import dgl
import torch
from dgl import EID, transform, NID
from dgl.dataloading.dataloader import _tensor_or_dict_to_numpy, _locate_eids_to_exclude, assign_block_eids
from torch.autograd.grad_mode import F


class BlockSampler(object):
    """Abstract class specifying the neighborhood sampling strategy for DGL data loaders.

    The main method for BlockSampler is :meth:`sample_blocks`,
    which generates a list of message flow graphs (MFGs) for a multi-layer GNN given a set of
    seed nodes to have their outputs computed.

    The default implementation of :meth:`sample_blocks` is
    to repeat :attr:`num_layers` times the following procedure from the last layer to the first
    layer:

    * Obtain a frontier.  The frontier is defined as a graph with the same nodes as the
      original graph but only the edges involved in message passing on the current layer.
      Customizable via :meth:`sample_frontier`.

    * Optionally, if the task is link prediction or edge classfication, remove edges
      connecting training node pairs.  If the graph is undirected, also remove the
      reverse edges.  This is controlled by the argument :attr:`exclude_eids` in
      :meth:`sample_blocks` method.

    * Convert the frontier into a MFG.

    * Optionally assign the IDs of the edges in the original graph selected in the first step
      to the MFG, controlled by the argument ``return_eids`` in
      :meth:`sample_blocks` method.

    * Prepend the MFG to the MFG list to be returned.

    All subclasses should override :meth:`sample_frontier`
    method while specifying the number of layers to sample in :attr:`num_layers` argument.

    Parameters
    ----------
    num_layers : int
        The number of layers to sample.
    return_eids : bool, default False
        Whether to return the edge IDs involved in message passing in the MFG.
        If True, the edge IDs will be stored as an edge feature named ``dgl.EID``.

    Notes
    -----
    For the concept of frontiers and MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    def __init__(self, num_layers, return_eids=False):
        self.num_layers = num_layers
        self.return_eids = return_eids

    def sample_frontier(self, block_id, g, seed_nodes):
        """Generate the frontier given the destination nodes.

        The subclasses should override this function.

        Parameters
        ----------
        block_id : int
            Represents which GNN layer the frontier is generated for.
        g : DGLGraph
            The original graph.
        seed_nodes : Tensor or dict[ntype, Tensor]
            The destination nodes by node type.

            If the graph only has one node type, one can just specify a single tensor
            of node IDs.

        Returns
        -------
        DGLGraph
            The frontier generated for the current layer.

        Notes
        -----
        For the concept of frontiers and MFGs, please refer to
        :ref:`User Guide Section 6 <guide-minibatch>` and
        :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
        """
        raise NotImplementedError

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """Generate the a list of MFGs given the destination nodes.

        Parameters
        ----------
        g : DGLGraph
            The original graph.
        seed_nodes : Tensor or dict[ntype, Tensor]
            The destination nodes by node type.

            If the graph only has one node type, one can just specify a single tensor
            of node IDs.
        exclude_eids : Tensor or dict[etype, Tensor]
            The edges to exclude from computation dependency.

        Returns
        -------
        list[DGLGraph]
            The MFGs generated for computing the multi-layer GNN output.

        Notes
        -----
        For the concept of frontiers and MFGs, please refer to
        :ref:`User Guide Section 6 <guide-minibatch>` and
        :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
        """
        blocks = []
        exclude_eids = (
            _tensor_or_dict_to_numpy(exclude_eids) if exclude_eids is not None else None)
        for block_id in reversed(range(self.num_layers)):
            frontier = self.sample_frontier(block_id, g, seed_nodes)

            # Removing edges from the frontier for link prediction training falls
            # into the category of frontier postprocessing
            if exclude_eids is not None:
                parent_eids = frontier.edata[EID]
                parent_eids_np = _tensor_or_dict_to_numpy(parent_eids)
                located_eids = _locate_eids_to_exclude(parent_eids_np, exclude_eids)
                if not isinstance(located_eids, Mapping):
                    # (BarclayII) If frontier already has a EID field and located_eids is empty,
                    # the returned graph will keep EID intact.  Otherwise, EID will change
                    # to the mapping from the new graph to the old frontier.
                    # So we need to test if located_eids is empty, and do the remapping ourselves.
                    if len(located_eids) > 0:
                        frontier = transform.remove_edges(
                            frontier, located_eids, store_ids=True)
                        frontier.edata[EID] = F.gather_row(parent_eids, frontier.edata[EID])
                else:
                    # (BarclayII) remove_edges only accepts removing one type of edges,
                    # so I need to keep track of the edge IDs left one by one.
                    new_eids = parent_eids.copy()
                    for k, v in located_eids.items():
                        if len(v) > 0:
                            frontier = transform.remove_edges(
                                frontier, v, etype=k, store_ids=True)
                            new_eids[k] = F.gather_row(parent_eids[k], frontier.edges[k].data[EID])
                    frontier.edata[EID] = new_eids

            block = transform.to_block(frontier, seed_nodes)

            if self.return_eids:
                assign_block_eids(block, frontier)

            seed_nodes = {ntype: block.srcnodes[ntype].data[NID] for ntype in block.srctypes}

            blocks.insert(0, block)
        return blocks


class ABC(metaclass=ABCMeta):
    """Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()


class Collator(ABC):
    """Abstract DGL collator for training GNNs on downstream tasks stochastically.

    Provides a :attr:`dataset` object containing the collection of all nodes or edges,
    as well as a :attr:`collate` method that combines a set of items from
    :attr:`dataset` and obtains the message flow graphs (MFGs).

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    @abstractproperty
    def dataset(self):
        """Returns the dataset object of the collator."""
        raise NotImplementedError

    @abstractmethod
    def collate(self, items):
        """Combines the items from the dataset object and obtains the list of MFGs.

        Parameters
        ----------
        items : list[str, int]
            The list of node or edge IDs or type-ID pairs.

        Notes
        -----
        For the concept of MFGs, please refer to
        :ref:`User Guide Section 6 <guide-minibatch>` and
        :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
        """
        raise NotImplementedError


class MultiLayerDropoutSampler(dgl.dataloading.BlockSampler):
    def __init__(self, p, num_layers):
        super().__init__(num_layers)
        self.p = p

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        # 获取 `seed_nodes` 的所有入边
        sg = dgl.in_subgraph(g, seed_nodes)

        new_edges_masks = {}
        # 遍历所有边的类型
        for etype in sg.canonical_etypes:
            edge_mask = torch.zeros(sg.number_of_edges(etype))
            edge_mask.bernoulli_(self.p)
            new_edges_masks[etype] = edge_mask.bool()

        # 返回一个与初始图有相同节点的图作为边界
        frontier = dgl.edge_subgraph(new_edges_masks, relabel_nodes=False)
        return frontier

    def __len__(self):
        return self.num_layers


class BioMIP_sampler(dgl.dataloading.BlockSampler):
    def __init__(self, n_interval, budgets: list, num_layers, p):
        super().__init__(num_layers)
        self.p = p

    def sample_frontier(self, block_id, g, seed_nodes):
        # 获取 `seed_nodes` 的所有入边
        sg = dgl.in_subgraph(g, seed_nodes)

        new_edges_masks = {}
        # 遍历所有边的类型
        for etype in sg.canonical_etypes:
            edge_mask = torch.zeros(sg.number_of_edges(etype))
            edge_mask.bernoulli_(self.p)
            new_edges_masks[etype] = edge_mask.bool()

        # 返回一个与初始图有相同节点的图作为边界
        frontier = dgl.edge_subgraph(new_edges_masks, relabel_nodes=False)
        return frontier

    def __len__(self):
        return self.num_layers
