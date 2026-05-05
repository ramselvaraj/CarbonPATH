import random
import json
import copy
from config import sram_selection_mode, print_info 
from collections import defaultdict

def _determine_protocols(hi_pkg_type, inter_conn, protocol_options):
    """
    Reusable helper function to determine 3D and 2.5D protocols.
    """
    protocol_3d = "na"
    protocol_2_5d = "na"

    # Filter available protocols from the input options for easier use
    opts_2_5d_std = [p for p in protocol_options if p == "ucie_std"]
    opts_2_5d_adv = [p for p in protocol_options if p in ["ucie_adv", "aib", "bow"]]
    opts_3d = [p for p in protocol_options if p == "ucie_3d"]

    #print("DEBUG Hi pkg type", hi_pkg_type)
    #print("DEBUG inter_cpmm", inter_conn)
    #print("DEBUG type inter_cpmm", type(inter_conn))
    
    if hi_pkg_type == "2d":
        return "na", "na"

    
    if hi_pkg_type == "2.5d":
        inter_conn_list = inter_conn if isinstance(inter_conn, list) else [inter_conn]

        conn_types = [
            c.get("connection_type", "na") if isinstance(c, dict)
            else c if isinstance(c, str)
            else "na"
            for c in inter_conn_list
        ]
        

        if "2.5d_rdl" in conn_types:
            protocol_2_5d = random.choice(opts_2_5d_std) if opts_2_5d_std else "na"
        elif any(ct in {"2.5d_emib", "2.5d_active", "2.5d_passive"} for ct in conn_types):
            protocol_2_5d = random.choice(opts_2_5d_adv) if opts_2_5d_adv else "na"
        else:
            protocol_2_5d = "na"
        return "na", protocol_2_5d


    if hi_pkg_type in ["3d", "2.5d_3d"]:
        if any("3d_" in c.get("connection_type", "") for c in inter_conn):
            protocol_3d = random.choice(opts_3d) if opts_3d else "na"
        
        if any(c.get("connection_type") == "2.5d_rdl" for c in inter_conn):
            protocol_2_5d = random.choice(opts_2_5d_std) if opts_2_5d_std else "na"
        elif any(c.get("connection_type") in ["2.5d_emib", "2.5d_active", "2.5d_passive"] for c in inter_conn):
            protocol_2_5d = random.choice(opts_2_5d_adv) if opts_2_5d_adv else "na"
    
        return protocol_3d, protocol_2_5d
        
    return protocol_3d, protocol_2_5d

def _validate_protocols(protocol_3d, protocol_2_5d, inter_conn):
   
    # Define allowed combinations for each protocol type
    allowed_3d = {
        "ucie_3d": ["3d_tsv", "3d_u_bump", "3d_hyb_bond"],
        "na": []
    }

    allowed_2_5d = {
        "ucie_std": ["2.5d_rdl"],
        "ucie_adv": ["2.5d_emib", "2.5d_active", "2.5d_passive"],
        "aib":       ["2.5d_emib", "2.5d_active", "2.5d_passive"],
        "bow":       ["2.5d_emib", "2.5d_active", "2.5d_passive"],
        "na": []
    }

    # Collect all connection types from interconnects
    #conn_types = [c.get("connection_type", "") for c in inter_conn]
    conn_types = [
    c["connection_type"] if isinstance(c, dict) else c
    for c in inter_conn
]

    # Validate 3D connections
    if protocol_3d != "na":
        if not any(conn in allowed_3d.get(protocol_3d, []) for conn in conn_types):
            return False  # Invalid 3D protocol usage

    # Validate 2.5D connections
    if protocol_2_5d != "na":
        if not any(conn in allowed_2_5d.get(protocol_2_5d, []) for conn in conn_types):
            return False  # Invalid 2.5D protocol usage

    return True

#Code v2 - with stacking rules 
class ChipletGenerator:
    def __init__(self, max_chiplets, sys_array_options, tech_node_options, sram_buf_options, area_power_func, sram_area_energy_func):
        self.max_chiplets, self.sys_array_options, self.tech_node_options, self.sram_buf_options, self.get_area_power, self.get_sram_area_energy = max_chiplets, sys_array_options, tech_node_options,  sram_buf_options, area_power_func, sram_area_energy_func

    def generate(self):
        num_chiplets = random.randint(1, self.max_chiplets)
        chiplets_dict = {}
        cg_single_sys_array_size = random.choice(self.sys_array_options) #Used for Chiplet Gym mode only
        for i in range(1, num_chiplets + 1):
            chiplet_name, selected_sys_array = f"Chiplet_{i}", random.choice(self.sys_array_options)
            selected_tech_node = random.choice(self.tech_node_options)
            
            # Select a valid SRAM buffer size for the chosen sys_array
            valid_sram_sizes = self.sram_buf_options.get(selected_sys_array, [256]) # Default if key is missing
            if sram_selection_mode == "random":
                sram_buf = random.choice(valid_sram_sizes)
            else: # sram_selection_mode is "max"
                # Select the largest SRAM buffer size
                sram_buf = max(valid_sram_sizes) if valid_sram_sizes else 256
            
            sram_area,_ = self.get_sram_area_energy(sram_buf,selected_tech_node)
            logic_area, power = self.get_area_power(selected_sys_array,selected_tech_node)
            area = logic_area + sram_area
            chiplets_dict[chiplet_name] = {"tech_node": selected_tech_node, "sys_array_size": selected_sys_array, "sram_buf": sram_buf, "area": area, "power": power}
        return chiplets_dict

class PackageGenerator:
    def __init__(self, chiplets_dict, inter_pkg_options, mem_pkg_options, protocol_options, same_3d_type=True, stack_diff_size=False):
        self.chiplets, self.num_chiplets = chiplets_dict, len(chiplets_dict)
        self.inter_pkg_options, self.mem_pkg_options = inter_pkg_options, mem_pkg_options
        self.protocol_options = protocol_options
        self.same_3d_type, self.stack_diff_size = same_3d_type, stack_diff_size
        self.opts_2_5d = [p for p in self.inter_pkg_options if p.startswith("2.5d")]
        self.opts_3d = [p for p in self.inter_pkg_options if p.startswith("3d")]

    def _find_valid_stack(self, available_chiplets, stack_size):
        # Rule: All chiplets in a stack must have the same sys_array_size and tech_node.
        if self.stack_diff_size is False:
            grouped_chiplets = defaultdict(list)
            for cid in available_chiplets:
                props = self.chiplets[cid]
                key = (props['sys_array_size'], props['tech_node'])
                grouped_chiplets[key].append(cid)
            # Find groups that are large enough for the required stack size
            valid_groups = [group for group in grouped_chiplets.values() if len(group) >= stack_size]
            if not valid_groups: return None
            chosen_group = random.choice(valid_groups)
            random.shuffle(chosen_group)
            return chosen_group[:stack_size]
        # Rule: Chiplets can have different sizes, but area must be <= the chiplet below it.
        else: # stack_diff_size is True
            potential_bases = list(available_chiplets); random.shuffle(potential_bases)
            for base_id in potential_bases:
                stack, remaining = [base_id], list(available_chiplets); remaining.remove(base_id)
                possible = True
                for _ in range(stack_size - 1):
                    last_area = self.chiplets[stack[-1]]['area']
                    # Find candidates that are small enough to place on top
                    candidates = [cid for cid in remaining if self.chiplets[cid]['area'] <= last_area]
                    if not candidates: possible = False; break
                    next_chiplet = random.choice(candidates)
                    stack.append(next_chiplet); remaining.remove(next_chiplet)
                if possible: return stack # Found a valid stack
            return None # No valid stack could be formed

    def _determine_hi_pkg_type(self):
        if self.num_chiplets == 1:
            return "2d"
        elif self.num_chiplets == 2:
            return random.choice(["2.5d", "3d"])
        else: # 3 or more
            return random.choice(["2.5d", "3d", "2.5d_3d"])
    
    def _determine_mem_pkg_type(self):
        ddr_options = [m for m in self.mem_pkg_options if 'ddr' in m.lower()]
        hbm_options = [m for m in self.mem_pkg_options if 'hbm' in m.lower()]

        
        if self.num_chiplets == 1:
            return random.choice(ddr_options) if ddr_options else random.choice(self.mem_pkg_options)
        else: # 2 or more chiplets
            combined_options = ddr_options + hbm_options
            return random.choice(combined_options) if combined_options else random.choice(self.mem_pkg_options)
        
        
    def _generate_3d_stack_connections(self, chiplet_ids_in_stack, stack_num):
        connections = []
        base_connection_type = random.choice(self.opts_3d) if self.opts_3d else "3d_tsv"
        for i, chiplet_id in enumerate(chiplet_ids_in_stack):
            from_chiplet, to_chiplet = chiplet_id, chiplet_ids_in_stack[i+1] if i+1 < len(chiplet_ids_in_stack) else "na"
            if i == 0: loc = f"stack{stack_num}_base"
            elif i == len(chiplet_ids_in_stack)-1: loc = f"stack{stack_num}_top"
            else: loc = f"stack{stack_num}_middle{i}"
            conn_type = base_connection_type if self.same_3d_type else random.choice(self.opts_3d)
            connections.append({"from": from_chiplet, "to": to_chiplet, "connection_type": conn_type, "loc": loc})
        return connections

    def _determine_inter_pkg_conn(self, hi_pkg_type):
        
        if hi_pkg_type == "2d": return "2d_na"
        if hi_pkg_type == "2.5d": return random.choice(self.opts_2_5d) if self.opts_2_5d else "2.5d_emib"
        # For a pure 3D package, all chiplets must form one valid stack.
        if hi_pkg_type == "3d":
            stack = self._find_valid_stack(list(self.chiplets.keys()), self.num_chiplets)
            if not stack: raise ValueError(f"Could not form a valid 3D stack of size {self.num_chiplets}.")
            return self._generate_3d_stack_connections(stack, 0)
        # For a hybrid package, we need at least one stack and one 2.5D chiplet.
        if hi_pkg_type == "2.5d_3d":
            if self.num_chiplets < 3: raise ValueError("2.5d_3d package requires at least 3 chiplets.")
            # The stack must leave at least one chiplet for the 2.5D connection.
            stack_size = random.randint(2, self.num_chiplets - 1)
            stack_chiplets = self._find_valid_stack(list(self.chiplets.keys()), stack_size)
            if not stack_chiplets: raise ValueError(f"Could not find a valid stack of size {stack_size}.")
            remaining_chiplets = [cid for cid in self.chiplets if cid not in stack_chiplets]
            stack_base = stack_chiplets[0]
            connections = self._generate_3d_stack_connections(stack_chiplets, 0)
            conn_2_5d_type = random.choice(self.opts_2_5d) if self.opts_2_5d else "2.5d_emib"
            for other_chiplet in remaining_chiplets:
                connections.append({"from": stack_base, "to": other_chiplet, "connection_type": conn_2_5d_type, "loc": "2.5d_chiplet"})
            return connections
        return "N/A"

    def _distribute_mem_channels(self, selected_mem_type, dont_use_random_channel_distribution=True): #TODO 3: Change use_random_channel_distribution to True to make this random
        chiplet_ids, num_chiplets = list(self.chiplets.keys()), self.num_chiplets
        
        
        selected_mem_type = selected_mem_type.lower()
        if dont_use_random_channel_distribution:
            # Use fixed values for each memory type
            if "ddr" in selected_mem_type:
                total_channels = 8
            elif "hbm" in selected_mem_type:
                total_channels = 16
            else:
                total_channels = 8
            
            total_channels = max(total_channels, num_chiplets)
        else:
            # Determine min and max channels based on memory type
            if "ddr" in selected_mem_type:
                min_ch, max_ch = 2, 8
            elif "hbm" in selected_mem_type:
                min_ch, max_ch = 8, 16
            else:
                min_ch, max_ch = 2, 8
            
            total_channels = (
                random.randint(max(min_ch, num_chiplets), max_ch)
                if max_ch >= num_chiplets else max_ch
            )
        
        #R total_channels = random.randint(max(min_ch, num_chiplets), max_ch) if max_ch >= num_chiplets else max_ch
        channels_dist, remaining_channels = [1] * num_chiplets, total_channels - num_chiplets
        if remaining_channels > 0:
            chiplet_areas = [self.chiplets[cid].get("area", 1.0) for cid in chiplet_ids]
            total_area = sum(chiplet_areas)
            if total_area > 0:
                proportions = [area / total_area for area in chiplet_areas]
                for _ in range(remaining_channels):
                    ideal_alloc = [(sum(channels_dist) + 1) * p for p in proportions]
                    errors = [ideal_alloc[i] - channels_dist[i] for i in range(num_chiplets)]
                    channels_dist[errors.index(max(errors))] += 1
            else:
                for i in range(remaining_channels): channels_dist[i % num_chiplets] += 1
        mem_conn_dict = {"mem_type": selected_mem_type}; mem_conn_dict.update(dict(zip(chiplet_ids, channels_dist)))
        return mem_conn_dict

    
    
    def generate(self, hi_pkg_type=None, mem_pkg_type=None, dont_use_random_channel_distribution=True, existing_pkg=None):
        
        if existing_pkg is None:
            print(f"[INFO] Running in gen_mode inside PkgGenerator") if print_info else None
            # Use provided package type or determine one randomly
            final_hi_pkg_type = hi_pkg_type if hi_pkg_type is not None else self._determine_hi_pkg_type()
            # Determine the interconnect based on the final package type
            inter_conn = self._determine_inter_pkg_conn(final_hi_pkg_type)
            # Use provided memory type or determine one randomly
            #R selected_mem_type = mem_pkg_type if mem_pkg_type is not None else random.choice(self.mem_pkg_options) if self.mem_pkg_options else "ddr5"
            selected_mem_type = mem_pkg_type if mem_pkg_type is not None else self._determine_mem_pkg_type()
            # Generate memory channel distribution
            mem_conn = self._distribute_mem_channels(selected_mem_type, dont_use_random_channel_distribution)
            # Determine the protocols based on the generated package
            protocol_3d, protocol_2_5d = _determine_protocols(final_hi_pkg_type, inter_conn, self.protocol_options)
            return {"HI_pkg_type": final_hi_pkg_type, "inter_pkg_conn": inter_conn, "protocol_3d": protocol_3d, "protocol_2.5d": protocol_2_5d, "mem_pkg_conn": mem_conn}
        else:
            # --- UPDATE MODE: Update an existing package ---
            print(f"[INFO] Running in update_mode inside PkgGenerator") if print_info else None
            updated_pkg = copy.deepcopy(existing_pkg)
            
            # 1. Rerun memory channel distribution as areas might have changed
            selected_mem_type = updated_pkg['mem_pkg_conn']['mem_type']
            updated_pkg['mem_pkg_conn'] = self._distribute_mem_channels(selected_mem_type, dont_use_random_channel_distribution)

            # 2. Validate and correct protocols based on existing interconnect
            updated_protocol_is_valid = _validate_protocols(protocol_3d=updated_pkg['protocol_3d'], protocol_2_5d=updated_pkg['protocol_2.5d'], inter_conn=updated_pkg['inter_pkg_conn'])
            if not updated_protocol_is_valid:
                p3d, p2_5d = _determine_protocols(hi_pkg_type, updated_pkg['inter_pkg_conn'], self.protocol_options)
                existing_pkg['protocol_3d'] = p3d
                existing_pkg['protocol_2.5d'] = p2_5d
            return updated_pkg
            

class WLMappingGenerator:
    
    def __init__(self):
        # Selection for if_split_k, ascending_order, dataflow
        self.mapping_options = {
            "chiplet_data_sharing_enabled": [0], 
            "if_splitting_k": [0, 1],
            "dataflow": ["ws", "os", "is"],
            "assign_workload_in_ascending_order": [0, 1],
            "static_tiling": [0], 
            "merge_tiles": [0] 
        }

    def generate(self):
        mapping_details = {}
        for key, options in self.mapping_options.items():
            chosen_value = random.choice(options)
            # The 'dataflow' key is special as it needs to be in a list
            if key == "dataflow":
                mapping_details[key] = [chosen_value]
            else:
                mapping_details[key] = chosen_value
        return mapping_details